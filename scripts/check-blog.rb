require 'nokogiri'
require 'pathname'
require 'uri'

root = Pathname.new(ARGV.fetch(0)).expand_path
fixture = ARGV.include?('--fixture')
count = 0
assert = lambda do |condition, message|
  raise message unless condition
  count += 1
end
read = ->(path) { Nokogiri::HTML5(root.join(path).read) }
index = read.call('blog/index.html')
assert.call(index.at_css('h1')&.text == 'Research notes', 'Missing blog index')
assert.call(index.at_css('link[rel="canonical"]')&.[]('href') == 'https://vanzll.github.io/blog/', 'Invalid canonical URL')

assert.call(!root.join('blog/diffusion-rl/index.html').exist?, 'Removed article leaked into production')
assert.call(!root.join('assets/blog/diffusion-rl').exist?, 'Removed manuscript figures leaked')
assert.call(!index.text.include?('What Actually Drives an Update?'), 'Draft leaked into blog index')
unless fixture
  assert.call(index.css('.post-row').empty?, 'Blog should remain empty')
end

home = read.call('index.html')
assert.call(home.at_css('#blog'), 'Missing homepage Blog section')
assert.call(home.at_css('.page__content a[href="/blog/"]'), 'Missing homepage blog link')
%w[educations services press-media].each do |id|
  assert.call(!home.at_css("[id='#{id}']"), "Removed section still present: #{id}")
  assert.call(home.css("a[href='/##{id}']").empty?, "Removed navigation still present: #{id}")
end
assert.call(home.css('.section-toggle-btn, .collapsible-content').empty?, 'Old hidden UI retained')
%w[news publications experiences].each do |section|
  assert.call(home.css(".#{section} .scroll-window[data-visible-items]").size == 1, "Missing scroll window: #{section}")
end
headings = home.css('.page__content h1').map(&:text)
assert.call(headings.index('🔥 News') < headings.index('Blog'), 'Blog must follow News')
assert.call(headings[headings.index('Blog') + 1] == '🎤 Invited Talks', 'Invited Talks must immediately follow Blog')
assert.call(!home.to_html.include?('toggleSection'), 'Old toggle script retained')
%w[publications-section internships-section invited-talks-section honors-section misc-section].each do |id|
  assert.call(home.at_css("[id='#{id}']"), "Lost retained section: #{id}")
end
assert.call(home.css('.exp-item').size == 4, 'Lost work experience')
ids = home.css('[id]').map { |node| node['id'] }
assert.call(ids.uniq.size == ids.size, 'Duplicate homepage anchors')

if fixture
  published = read.call('blog/test-fixture/index.html')
  assert.call(published.at_css('head meta[name="citation_title"]'), 'Published citation metadata must be in head')
  assert.call(published.at_css('meta[name="citation_author"]')&.[]('content') == 'Test Author', 'Missing citation author')
  assert.call(published.at_css('#citation-bibtex')&.text&.include?('@misc'), 'Missing published BibTeX')
  assert.call(published.at_css('[data-copy="citation-bibtex"]'), 'Missing citation copy button')
  assert.call(published.css('meta[name="robots"]').empty?, 'Published post incorrectly noindex')
end

docs = [index, home]
docs.each do |doc|
  doc.css('link[href], script[src], img[src]').each do |node|
    url = node['href'] || node['src']
    next unless url.start_with?('/') && !url.start_with?('//')
    path = URI::DEFAULT_PARSER.unescape(url.split(/[?#]/).first.delete_prefix('/'))
    assert.call(root.join(path).file?, "Missing local asset: #{url}")
  end
end
assert.call(!root.join('scripts').exist?, 'Developer scripts should not be deployed')
puts "Website checks passed: #{count} (fixture=#{fixture})"
