require 'digest'
require 'fileutils'
require 'json'
require 'nokogiri'
require 'pathname'
require 'time'

abort 'Use --publish after prose and figure review.' unless ARGV == ['--publish']
root = Pathname.new(__dir__).parent
private_prefix = '/_draft_assets/diffusion-rl/'
public_prefix = '/assets/blog/diffusion-rl/'
sections = {}
snapshots = {}

%w[zh en].each do |language|
  draft = root.join("_includes/blog-drafts/diffusion-rl.#{language}.md").read
  public_path = root.join("_includes/research-notes/diffusion-rl.#{language}.md")
  public_source = public_path.read
  section_pattern = /<h3 id="#{language}-detail-11">.*?(?=<h2 id="#{language}-recipes")/m
  note_start = language == 'zh' ? '- 图9' : '- Figures 9'
  note_end = language == 'zh' ? '- 待补实验' : '- Proposed experiments'
  notes_pattern = /#{Regexp.escape(note_start)}.*?(?=#{Regexp.escape(note_end)})/m
  [section_pattern, notes_pattern].each do |pattern|
    abort "Missing or duplicate boundary for #{language}" unless draft.scan(pattern).size == 1 && public_source.scan(pattern).size == 1
  end
  sections[language] = draft[section_pattern]
  new_section = sections[language].gsub(private_prefix, public_prefix)
  new_notes = draft[notes_pattern].gsub(private_prefix, public_prefix)
  updated = public_source.sub(section_pattern) { new_section }.sub(notes_pattern) { new_notes }
  # Everything outside the requested section and its data notes stays byte-identical.
  strip = ->(source) { source.sub(section_pattern, '').sub(notes_pattern, '') }
  abort "Out-of-scope publication change: #{language}" unless strip.call(updated) == strip.call(public_source)
  abort "Private path in #{language}" if updated.match?(%r{(?:/Users/|/m2v_intern|127\.0\.0\.1|localhost|_draft_assets)})
  snapshots[language] = { path: public_path, source: updated }
end

documents = sections.transform_values { |source| Nokogiri::HTML.fragment(source) }
equations = documents.transform_values { |doc| doc.css('[data-equation]').map { |node| [node['data-equation'], node.text.strip] } }
figures = documents.transform_values { |doc| doc.css('figure img').map { |node| node['src'] } }
abort 'Section equations differ between languages.' unless equations['zh'] == equations['en']
abort 'Section figures differ between languages.' unless figures['zh'] == figures['en']

asset_names = figures['zh'].map { |path| File.basename(path) }
%w[force_mechanisms force_direction_control force_stability].each do |name|
  %w[pdf csv json].each { |extension| asset_names << "#{name}.#{extension}" }
end
%w[force_mechanisms force_direction_control].each { |name| asset_names << "#{name}_windows.csv" }
asset_names.uniq.each do |name|
  # The renderer emits sanitized public JSON separately from private snapshots.
  directory = File.extname(name) == '.json' ? 'assets/blog/diffusion-rl' : '_draft_assets/diffusion-rl'
  source = root.join(directory, name)
  abort "Missing reviewed asset: #{name}" unless source.file?
  if %w[.csv .json].include?(source.extname)
    abort "Private path in asset #{name}" if source.read.match?(%r{(?:/Users/|/m2v_intern|127\.0\.0\.1|localhost)})
  end
end
asset_names.uniq.each do |name|
  next if File.extname(name) == '.json'
  FileUtils.cp(root.join('_draft_assets/diffusion-rl', name), root.join('assets/blog/diffusion-rl', name))
end
snapshots.each_value { |snapshot| snapshot[:path].write(snapshot[:source]) }

manifest_path = root.join('docs/diffusion-rl-publication.json')
manifest = JSON.parse(manifest_path.read)
manifest['base_full_draft_sha256'] = manifest.delete('source_sha256') if manifest.key?('source_sha256')
manifest['published_at'] = Time.now.utc.iso8601
manifest['last_update_scope'] = 'Section 3.3 and its figure data notes only; remaining public text preserved.'
manifest['public_sha256'] = snapshots.transform_values { |snapshot| Digest::SHA256.hexdigest(snapshot[:source]) }
manifest['section_3_3_source_sha256'] = sections.transform_values { |source| Digest::SHA256.hexdigest(source) }
manifest['figures'] = Nokogiri::HTML.fragment(snapshots['zh'][:source]).css('figure img').to_h do |image|
  path = root.join(image['src'].delete_prefix('/'))
  [path.basename.to_s, Digest::SHA256.file(path).hexdigest]
end
manifest_path.write(JSON.pretty_generate(manifest) + "\n")
puts "Updated bilingual Section 3.3 only (#{figures['zh'].size} figures); unlisted page metadata unchanged."
