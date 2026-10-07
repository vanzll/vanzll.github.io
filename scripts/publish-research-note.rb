require 'date'
require 'digest'
require 'fileutils'
require 'json'
require 'nokogiri'
require 'pathname'
require 'time'
require 'yaml'

abort 'Use --publish after synchronizing the English translation.' unless ARGV == ['--publish']
root = Pathname.new(__dir__).parent
draft = root.join('_drafts/understanding-diffusion-rl.md').read
front_matter, body = draft.split(/^---\s*$\n/, 3).drop(1)
metadata = YAML.safe_load(front_matter)
sources = %w[zh en].to_h do |language|
  [language, root.join("_includes/blog-drafts/diffusion-rl.#{language}.md").read]
end
documents = sources.transform_values { |source| Nokogiri::HTML.fragment(source) }
equations = documents.transform_values do |document|
  document.css('[data-equation]').map { |node| [node['data-equation'], node.text.strip] }
end
abort 'Chinese and English mathematical anchors differ.' unless equations['zh'] == equations['en']
figures = documents.transform_values { |document| document.css('figure img').map { |image| image['src'] } }
abort 'Chinese and English figure sequences differ.' unless figures['zh'] == figures['en']

asset_prefix = '/_draft_assets/diffusion-rl/'
public_prefix = '/assets/blog/diffusion-rl/'
sources.each_value do |source|
  abort 'Private paths or local services found in manuscript.' if source.match?(%r{(?:/Users/|/m2v_intern|127\.0\.0\.1|localhost|review\.js)})
end
figures['zh'].uniq.each do |path|
  abort "Unexpected figure path: #{path}" unless path.start_with?(asset_prefix) && File.basename(path) == path.delete_prefix(asset_prefix)
  source = root.join(path.delete_prefix('/'))
  destination = root.join(path.sub(asset_prefix, public_prefix).delete_prefix('/'))
  FileUtils.mkdir_p(destination.dirname)
  FileUtils.cp(source, destination)
end

# A public snapshot leaves the local editable draft and review history private.
sources.each do |language, source|
  destination = root.join("_includes/research-notes/diffusion-rl.#{language}.md")
  FileUtils.mkdir_p(destination.dirname)
  destination.write(source.gsub(asset_prefix, public_prefix))
end
metadata['draft'] = false
metadata['unlisted'] = true
metadata['sitemap'] = false
metadata['date'] = '2026-10-07 12:00:00 +0800'
metadata['description_en'] = 'From output-space forces to shared-parameter updates: understanding the efficiency and stability of diffusion RL.'
metadata['image'] = public_prefix + 'force_overview.png'
metadata['bibtex'] = <<~BIBTEX
  @misc{wan2026continuousgenerativepolicy,
    author = {Zhenglin Wan},
    title = {Understanding Policy Optimization in Continuous-Time Generative Processes: Phenomena, Insights, and Recipes},
    year = {2026},
    howpublished = {Research note},
    url = {https://vanzll.github.io/blog/diffusion-rl/}
  }
BIBTEX
post = root.join('_pages/diffusion-rl.md')
FileUtils.mkdir_p(post.dirname)
post.write(metadata.to_yaml + "---\n" + body.gsub('blog-drafts/', 'research-notes/'))

manifest = {
  'published_at' => Time.now.utc.iso8601,
  'url' => 'https://vanzll.github.io/blog/diffusion-rl/',
  'visibility' => 'unlisted',
  'source_sha256' => sources.transform_values { |source| Digest::SHA256.hexdigest(source) },
  'figures' => figures['zh'].uniq.to_h do |path|
    [File.basename(path), Digest::SHA256.file(root.join(path.delete_prefix('/'))).hexdigest]
  end
}
root.join('docs/diffusion-rl-publication.json').write(JSON.pretty_generate(manifest) + "\n")
puts "Prepared bilingual public snapshot: #{post.relative_path_from(root)} (#{figures['zh'].uniq.size} shared figures)"
