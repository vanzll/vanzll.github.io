(function () {
  function resizeWindows() {
    document.querySelectorAll('.scroll-window[data-visible-items]').forEach(function (container) {
      var item = container.querySelector('ul > li');
      if (!item) return;
      var style = window.getComputedStyle(item);
      var height = item.getBoundingClientRect().height;
      var margins = (parseFloat(style.marginTop) || 0) + (parseFloat(style.marginBottom) || 0);
      container.style.maxHeight = ((height + margins) * Number(container.dataset.visibleItems)) + 'px';
    });
  }

  resizeWindows();
  window.addEventListener('load', resizeWindows, { once: true });
  var resizeTimeout;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(resizeWindows, 150);
  });
})();
