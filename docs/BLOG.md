# Research blog

This Jekyll template reconstructs the layout of Thinking Machines Lab's [On-Policy Distillation blog](https://thinkingmachines.ai/blog/on-policy-distillation/): centered serif title and byline, a centered reading column, and a restrained left-margin TOC. The personal homepage has a Blog section linking to published research notes.

This is an independently implemented visual adaptation, not an official TML template. TML branding and proprietary webfont files are not copied. On this Mac it uses the installed Iowan Old Style font; other systems fall back to Palatino or Georgia. Navigation uses system sans-serif instead of licensed GT America. No external runtime request is needed.

- Index: `_pages/blog.html`, public route `/blog/`.
- Template: `_layouts/research.html`, `_layouts/research-post.html`.
- Styling and interactions: `assets/css/research.css`, `assets/js/research.js`.
- Future article images: put approved assets under `assets/blog/`.
- Math: self-hosted KaTeX 0.16.22, including its fonts and MIT license. No external runtime request is needed for article rendering.

## Preview

Use Ruby 3.3 and Bundler. The build is aligned with GitHub Pages 232 (Jekyll 3.10); the previous lockfile used 2021 dependencies incompatible with modern Ruby. On this Mac:

```sh
export PATH="/opt/homebrew/opt/ruby@3.3/bin:$PATH"
export BUNDLE_VERSION=system
bundle install
bundle exec jekyll serve --host 127.0.0.1 --port 4000
```

The blog appears at `http://127.0.0.1:4000/blog/`. For private local drafts, create a file in `_drafts/` and preview with `--drafts`. Ordinary builds do not include drafts. Draft pages carry `noindex,nofollow` and omit citation metadata. Do not upload a draft-preview build as the public website.

Drafts are not private if their source is pushed to a public repository. Keep unapproved manuscripts local or in a private repository.

## Add or publish an article

Create an article with `layout: research-post`, title, description, authors, TOC, and permalink in its YAML front matter. Use plain Markdown for prose. For equations, use a raw HTML math container, so Kramdown does not interpret TeX escapes:

```html
<div class="math">\[g_i = J_i^{\mathsf T} f_i\]</div>
```

An insight block:

```html
<div class="insight" markdown="1">
**The main observation.**

Explain what the reader should take away.
</div>
```

To publish, move the file to `_posts/YYYY-MM-DD-slug.md`, set `draft: false`, and use its actual publication date. Review authorship, claims, figure provenance, and permissions before doing so. Add `bibtex: |` with a citation for the blog itself; do not invent a DOI. Add `doi`, `paper_url`, or `pdf` only after those resources exist. The companion paper and blog can have separate citations.

Use stable permalinks. Preserve previously published snapshots and archive releases separately when a permanent citation version is needed. Hosting on GitHub Pages is not a perpetual archival guarantee.

To set the opening concept figure, add `cover`, `cover_alt`, `cover_width`, `cover_height`, and optionally `cover_caption` to the article front matter. Keep its correct intrinsic dimensions. The cover is placed between the byline and the introduction; do not duplicate it inside the article body. The desktop TOC stays outside the centered text column and is hidden on narrower screens, matching the reference layout.

## Checks

```sh
bundle exec jekyll build --destination /tmp/research-blog-public
bundle exec ruby scripts/check-blog.rb /tmp/research-blog-public
```

References: https://distill.pub/guide/ and https://thinkingmachines.ai/blog/on-policy-distillation/ .

## Bilingual publication snapshot

The Chinese draft is the editorial master. After synchronizing its English
translation and receiving explicit publication approval, run:

```sh
bundle exec ruby scripts/publish-research-note.rb --publish
bundle exec jekyll build --destination /tmp/research-blog-public
bundle exec ruby scripts/check-published-note.rb /tmp/research-blog-public
```

The public note lives at `/blog/diffusion-rl/`, with one shared figure set and an
in-page Chinese/English switch. It is an unlisted page, not a feed post, and uses `_includes/research-notes/` and
`assets/blog/diffusion-rl/`. Private drafts, author reviews, editing services,
raw probe archives and local machine paths are not published. The local review
draft remains separate; exclude this public page from the draft preview config
to avoid two pages claiming the same permalink. Publication fingerprints are
recorded in `docs/diffusion-rl-publication.json`; later local edits do not silently
change the published snapshot.

The unlisted page has `sitemap: false` and `noindex,nofollow`. It is absent from
the homepage, article index, RSS and sitemap, but is not password-protected:
anyone with the direct URL can read it, and its source is in the public repository.

Vendored icons: Lucide 0.468.0 (`copy.svg`, `x.svg`, `menu.svg`), license retained in `assets/vendor/lucide/LICENSE`.
