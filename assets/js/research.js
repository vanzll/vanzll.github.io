(() => {
  const menu = document.querySelector('.nav-toggle');
  const navigation = document.getElementById('research-nav');
  const closeMenu = () => {
    if (!menu || !navigation) return;
    menu.setAttribute('aria-expanded', 'false');
    navigation.dataset.open = 'false';
  };
  if (menu && navigation) {
    menu.addEventListener('click', () => {
      const open = menu.getAttribute('aria-expanded') !== 'true';
      menu.setAttribute('aria-expanded', String(open));
      navigation.dataset.open = String(open);
    });
    navigation.addEventListener('click', event => {
      if (event.target.closest('a')) closeMenu();
    });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && menu.getAttribute('aria-expanded') === 'true') {
        closeMenu();
        menu.focus();
      }
    });
    window.matchMedia('(max-width: 760px)').addEventListener('change', closeMenu);
  }
  const article = document.querySelector('.article-body');
  if (!article) return;

  if (article.dataset.math === 'true' && typeof renderMathInElement === 'function') {
    renderMathInElement(article, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false },
      ],
      throwOnError: false,
    });
  }

  article.querySelectorAll('h2[id], h3[id]').forEach(heading => {
    const anchor = document.createElement('a');
    anchor.className = 'section-anchor';
    anchor.href = `#${heading.id}`;
    anchor.setAttribute('aria-label', `Link to section: ${heading.textContent}`);
    anchor.textContent = '#';
    heading.prepend(anchor);
  });
  const links = [...document.querySelectorAll('.article-toc a')];
  const headings = links.map(link => document.getElementById(link.hash.slice(1))).filter(Boolean);
  if (headings.length) {
    let pending = false;
    const updateToc = () => {
      let current = headings[0];
      for (const heading of headings) {
        if (heading.getBoundingClientRect().top <= 110) current = heading;
      }
      links.forEach(link => {
        if (link.hash === `#${current.id}`) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
      pending = false;
    };
    const scheduleToc = () => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(updateToc);
    };
    window.addEventListener('scroll', scheduleToc, { passive: true });
    window.addEventListener('resize', scheduleToc);
    updateToc();
  }

  const dialog = document.querySelector('.figure-dialog');
  if (dialog && typeof dialog.showModal === 'function') {
    document.querySelectorAll('.research-article figure img').forEach(image => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'figure-expand';
      button.title = 'Enlarge figure';
      button.setAttribute('aria-label', `Enlarge figure: ${image.alt}`);
      image.before(button);
      button.append(image);
      button.addEventListener('click', () => {
        const enlarged = dialog.querySelector('.figure-dialog-image');
        enlarged.src = image.currentSrc || image.src;
        enlarged.alt = image.alt;
        dialog.showModal();
      });
    });
    dialog.querySelector('.figure-close').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => {
      if (event.target === dialog) dialog.close();
    });
  }

  document.querySelectorAll('[data-copy]').forEach(button => {
    button.addEventListener('click', async () => {
      const text = document.getElementById(button.dataset.copy)?.textContent;
      const status = document.querySelector('.copy-status');
      if (!text || !status) return;
      try {
        await navigator.clipboard.writeText(text);
        status.textContent = 'Citation copied.';
      } catch {
        status.textContent = 'Copy unavailable. Select the BibTeX text below.';
      }
    });
  });
})();
