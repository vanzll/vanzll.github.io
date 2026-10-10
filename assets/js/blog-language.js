(() => {
  const button = document.querySelector('[data-language-toggle]');
  const panels = [...document.querySelectorAll('[data-language-panel]')];
  if (!button || panels.length !== 2) return;

  let language = panels.find(panel => !panel.hidden)?.dataset.languagePanel || 'en';
  button.addEventListener('click', () => {
    const visible = panels.find(panel => !panel.hidden);
    const section = [...visible.querySelectorAll('h2[data-section-key]')]
      .filter(heading => heading.getBoundingClientRect().top <= 110).pop();
    const previousOffset = section?.getBoundingClientRect().top;
    language = language === 'zh' ? 'en' : 'zh';
    panels.forEach(panel => { panel.hidden = panel.dataset.languagePanel !== language; });
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-i18n]').forEach(element => {
      element.textContent = element.dataset[language];
    });
    document.querySelectorAll('.article-toc a[data-section-key]').forEach(link => {
      link.setAttribute('href', `#${language}-${link.dataset.sectionKey}`);
    });
    button.textContent = language === 'zh' ? 'English' : '中文';
    button.setAttribute('aria-label', language === 'zh' ? 'Switch to English' : '切换为中文');
    document.title = `${document.querySelector('.article-header h1').textContent} | Zhenglin Wan`;
    if (section) {
      const destination = document.getElementById(`${language}-${section.dataset.sectionKey}`);
      window.scrollBy({ top: destination.getBoundingClientRect().top - previousOffset, behavior: 'instant' });
    }
    document.dispatchEvent(new CustomEvent('blog:languagechange', { detail: { language } }));
  });
})();
