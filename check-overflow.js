const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080 });

  const filePath = path.resolve(__dirname, 'presentation.html');
  await page.goto(`file://${filePath}`);
  await page.waitForSelector('.reveal.ready');

  const totalSlides = await page.evaluate(() => Reveal.getTotalSlides());
  console.log(`Checking ${totalSlides} slides for overflow...\n`);

  const issues = [];

  for (let i = 0; i < totalSlides; i++) {
    await page.evaluate((idx) => Reveal.slide(idx), i);
    await new Promise(r => setTimeout(r, 200));

    const slideInfo = await page.evaluate(() => {
      const indices = Reveal.getIndices();
      return { h: indices.h, v: indices.v };
    });

    const overflows = await page.evaluate(() => {
      const results = [];
      const selectors = '.present p, .present li, .present h1, .present h2, .present h3, .present .card, .present pre, .present .kpi, .present td, .present th';

      document.querySelectorAll(selectors).forEach(el => {
        const dominated = el.closest('.present') !== document.querySelector('section.present');
        if (dominated) return;

        const style = getComputedStyle(el);
        const isScrollable = style.overflowY === 'auto' || style.overflowY === 'scroll';

        const vDiff = el.scrollHeight - el.clientHeight;
        const hDiff = el.scrollWidth - el.clientWidth;
        // Use 20px threshold to filter line-height false positives
        if (!isScrollable && (vDiff > 20 || hDiff > 20)) {
          results.push({
            tag: el.tagName,
            class: el.className,
            text: el.textContent.trim().slice(0, 60).replace(/\s+/g, ' '),
            overflow: {
              vertical: vDiff > 20 ? `${vDiff}px` : false,
              horizontal: hDiff > 20 ? `${hDiff}px` : false
            }
          });
        }
      });
      return results;
    });

    if (overflows.length) {
      issues.push({ slide: i + 1, indices: slideInfo, overflows });
    }
  }

  if (issues.length === 0) {
    console.log('No overflow issues detected.');
  } else {
    console.log(`Found overflow issues on ${issues.length} slide(s):\n`);
    issues.forEach(({ slide, indices, overflows }) => {
      console.log(`Slide ${slide} (h:${indices.h}, v:${indices.v}):`);
      overflows.forEach(o => {
        const dir = [o.overflow.vertical && 'vertical', o.overflow.horizontal && 'horizontal'].filter(Boolean).join(', ');
        console.log(`  - <${o.tag.toLowerCase()}${o.class ? '.' + o.class.split(' ')[0] : ''}> [${dir}]`);
        console.log(`    "${o.text}..."`);
      });
      console.log('');
    });
  }

  await browser.close();
})();
