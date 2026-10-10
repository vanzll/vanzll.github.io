require 'nokogiri'
require 'pathname'
require 'uri'

root = Pathname.new(ARGV.fetch(0))
read = ->(path) { Nokogiri::HTML5(root.join(path).read) }
def check(condition, message)
  raise message unless condition
end
article = read.call('blog/diffusion-rl/index.html')
check(article.at_css('link[rel="canonical"]')['href'] == 'https://vanzll.github.io/blog/diffusion-rl/', 'Incorrect public canonical URL')
check(article.css('meta[name="robots"]').any? { |node| node['content'].include?('noindex') }, 'Unlisted note must discourage indexing')
check(article.at_css('meta[name="citation_title"]') && article.at_css('#citation-bibtex'), 'Missing public citation')
check(article.at_css('[data-language-toggle]')&.text == 'English', 'Missing language switch')
panels = article.css('[data-language-panel]')
check(panels.size == 2 && !panels[0].key?('hidden') && panels[1].key?('hidden'), 'Invalid bilingual panel state')
equations = panels.map do |panel|
  panel.css('[data-equation]').map do |node|
    text = node.text.strip.gsub('单样本响应率', 'Per-state response rate')
               .gsub('聚合保留率', 'Aggregation retention')
    [node['data-equation'], text]
  end
end
check(equations[0] == equations[1] && !equations[0].empty?, 'Translated equations differ')
check(panels[0].css('figure img').map { |node| node['src'] } == panels[1].css('figure img').map { |node| node['src'] }, 'Translated figures differ')
check(panels[0].css('h2, h3, h4').map { |node| node['id'].sub(/^zh-/, '') } == panels[1].css('h2, h3, h4').map { |node| node['id'].sub(/^en-/, '') }, 'Translated section structure differs')
check(article.css('[data-section-key="evidence"]').empty?, 'Data notes must not be displayed')
check(article.css('[data-equation="mirror-loss"], [data-equation="covariance-identity"]').empty?, 'Duplicate appendix derivations must not be displayed')
%w[reward-tilt baseline transfer restoration].each do |name|
  check(panels.all? { |panel| panel.at_css("[data-equation='#{name}']") }, "Missing unique appendix formula: #{name}")
end
article.css('.article-toc a[href]').each do |link|
  check(article.at_css("[id='#{link['href'].delete_prefix('#')}']"), 'Broken article TOC')
end
ids = article.css('[id]').map { |node| node['id'] }
check(ids.uniq.size == ids.size, 'Duplicate article anchors')
%w[index.html blog/index.html feed.xml sitemap.xml].each do |path|
  check(!root.join(path).read.include?('/blog/diffusion-rl/'), "Unlisted note exposed in #{path}")
end
[article, read.call('index.html'), read.call('blog/index.html')].each do |document|
  check(!document.to_html.match?(%r{(?:_draft_assets|review\.js|review-toggle|localhost|127\.0\.0\.1|/Users/|/m2v_intern)}), 'Private review or local path leaked')
  document.css('link[href], script[src], img[src]').each do |node|
    url = node['href'] || node['src']
    next unless url.start_with?('/') && !url.start_with?('//')
    path = URI::DEFAULT_PARSER.unescape(url.split(/[?#]/).first.delete_prefix('/'))
    check(root.join(path).file?, "Missing public asset: #{url}")
  end
end
%w[reviews scripts _draft_assets _drafts].each do |path|
  check(!root.join(path).exist?, "Private directory deployed: #{path}")
end
puts "Public note checks passed: #{equations[0].size} mathematical anchors, #{panels[0].css('figure img').size} shared figures"
