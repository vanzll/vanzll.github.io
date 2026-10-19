(() => {
  const article = document.querySelector('.article-body');
  if (!article) return;

  function prepareContent(root) {
  if (article.dataset.math === 'true' && typeof renderMathInElement === 'function') {
    renderMathInElement(root, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false },
      ],
      throwOnError: false,
    });
  }

  const equationPanels = article.querySelectorAll('[data-language-panel]');
  (equationPanels.length ? [...equationPanels] : [article]).forEach(panel => {
    const language = panel.dataset.languagePanel || 'article';
    const equations = new Map();
    panel.querySelectorAll('div.math[data-equation]').forEach((equation, index) => {
      equation.id = `eq-${language}-${equation.dataset.equation}`;
      equation.querySelectorAll('.equation-number').forEach(node => node.remove());
      const number = document.createElement('a');
      number.className = 'equation-number';
      number.href = `#${equation.id}`;
      number.textContent = `(${index + 1})`;
      number.setAttribute('aria-label', `${language === 'zh' ? '公式' : 'Equation'} ${index + 1}`);
      equation.append(number);
      equations.set(equation.dataset.equation, { id: equation.id, number: index + 1 });
    });
    panel.querySelectorAll('[data-equation-ref]').forEach(reference => {
      const equation = equations.get(reference.dataset.equationRef);
      if (!equation) return;
      reference.href = `#${equation.id}`;
      if (!reference.textContent.trim()) reference.textContent = `(${equation.number})`;
    });
  });

  root.querySelectorAll('h2[id], h3[id], h4[id]').forEach(heading => {
    if (heading.querySelector('.section-anchor')) return;
    const anchor = document.createElement('a');
    anchor.className = 'section-anchor';
    anchor.href = `#${heading.id}`;
    anchor.setAttribute('aria-label', `Link to section: ${heading.textContent}`);
    anchor.textContent = '#';
    heading.prepend(anchor);
  });
  const figureRoot = root === article ? document.querySelector('.research-article') || root : root;
  figureRoot.querySelectorAll('figure img').forEach(image => {
    if (image.closest('.figure-expand') || image.closest('button')) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'figure-expand';
    button.title = 'Enlarge figure';
    button.setAttribute('aria-label', `Enlarge figure: ${image.alt}`);
    image.before(button);
    button.append(image);
  });
  }
  prepareContent(article);
  document.addEventListener('blog:contentchange', event => prepareContent(event.detail.panel));

  const revealLinkedDerivation = () => {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); }
    catch { return; }
    const target = document.getElementById(id);
    if (!target || !article.contains(target)) return;
    let opened = false;
    for (let parent = target.parentElement; parent && parent !== article; parent = parent.parentElement) {
      if (parent.tagName === 'DETAILS' && !parent.open) {
        parent.open = true;
        opened = true;
      }
    }
    if (opened) requestAnimationFrame(() => target.scrollIntoView({block: 'center'}));
  };
  window.addEventListener('hashchange', revealLinkedDerivation);
  revealLinkedDerivation();

  const links = [...document.querySelectorAll('.article-toc a')];
  let headings = links.map(link => document.getElementById(link.hash.slice(1))).filter(Boolean);
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
    const refreshHeadings = () => {
      headings = links.map(link => document.getElementById(link.hash.slice(1))).filter(Boolean);
      scheduleToc();
    };
    document.addEventListener('blog:languagechange', refreshHeadings);
    document.addEventListener('blog:contentchange', refreshHeadings);
    updateToc();
  }

  const dialog = document.querySelector('.figure-dialog');
  if (dialog && typeof dialog.showModal === 'function') {
    (document.querySelector('.research-article') || article).addEventListener('click', event => {
      const button = event.target.closest('.figure-expand');
      if (!button) return;
      const image = button.querySelector('img');
      if (image) {
        const enlarged = dialog.querySelector('.figure-dialog-image');
        enlarged.src = image.currentSrc || image.src;
        enlarged.alt = image.alt;
        dialog.showModal();
      }
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
