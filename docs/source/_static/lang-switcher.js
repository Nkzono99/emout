/**
 * Language switcher for emout docs.
 *
 * - Adds a Japanese/English dropdown next to furo's dark-mode toggle.
 * - Filters the sidebar toctree to show only the active language.
 * - Persists the choice in localStorage.
 *
 * Page naming convention:
 *   - Guide:  quickstart.ja.html (JA) / quickstart.html (EN mirror)
 *   - Index:  index.html (JA)         / index.en.html (EN mirror)
 */
const STORAGE_KEY = "emout-docs-lang";

// Early redirect: if the user previously chose a language, send them to the
// matching counterpart before the page paints. Runs at script-load time so
// there is no flash of the wrong language.
(function redirectToPreferredLang() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return;
    const path = window.location.pathname;
    const isGuide = path.includes("/guide/");
    const isGuideJa = isGuide && path.endsWith(".ja.html");
    const isGuideEn = isGuide && !isGuideJa;
    const isIndexEn = /\/index\.en\.html$/.test(path);
    const isIndexJa =
      !isGuide && !isIndexEn && (path.endsWith("/") || /\/index\.html$/.test(path));

    let target = null;
    if (stored === "en" && isGuideJa) {
      target = path.replace(/\.ja\.html$/, ".html");
    } else if (stored === "ja" && isGuideEn) {
      target = path.replace(/\.html$/, ".ja.html");
    } else if (stored === "en" && isIndexJa) {
      target = path.endsWith("/")
        ? path + "index.en.html"
        : path.replace(/\/index\.html$/, "/index.en.html");
    } else if (stored === "ja" && isIndexEn) {
      target = path.replace(/\/index\.en\.html$/, "/index.html");
    }
    if (target && target !== path) {
      window.location.replace(target + window.location.hash);
    }
  } catch (_e) {
    // Ignore — redirect is best-effort.
  }
})();

document.addEventListener("DOMContentLoaded", () => {
  const path = window.location.pathname;

  const isGuide = path.includes("/guide/");
  const isGuideJa = isGuide && path.endsWith(".ja.html");
  const isGuideEn = isGuide && !isGuideJa;

  const isIndexEn = /\/index\.en\.html$/.test(path);
  const isIndex = !isGuide && (path.endsWith("/") || /\/index(\.en)?\.html$/.test(path));
  const isIndexJa = isIndex && !isIndexEn;

  // The current page's language, if it can be determined from the URL.
  const pageLang = (isGuideJa || isIndexJa)
    ? "ja"
    : (isGuideEn || isIndexEn ? "en" : null);

  // Prefer the page's own language over localStorage so that the selector
  // reflects what the user is actually looking at. Non-localized pages
  // (e.g. API reference) fall back to the stored preference.
  const lang = pageLang || localStorage.getItem(STORAGE_KEY) || "ja";

  // Keep localStorage in sync when landing on a localized page directly.
  if (pageLang) {
    localStorage.setItem(STORAGE_KEY, pageLang);
  }

  // Compute the counterpart URL in the target language (null if unavailable).
  function counterpartUrl(targetLang) {
    if (isGuide) {
      if (targetLang === "ja" && isGuideEn) {
        return path.replace(/\.html$/, ".ja.html");
      }
      if (targetLang === "en" && isGuideJa) {
        return path.replace(/\.ja\.html$/, ".html");
      }
      return null;
    }
    if (isIndex) {
      if (targetLang === "en" && isIndexJa) {
        if (path.endsWith("/")) return path + "index.en.html";
        return path.replace(/\/index\.html$/, "/index.en.html");
      }
      if (targetLang === "ja" && isIndexEn) {
        return path.replace(/\/index\.en\.html$/, "/index.html");
      }
      return null;
    }
    return null;
  }

  // --- Sidebar filtering ---
  function filterSidebar(activeLang) {
    const sidebarLinks = document.querySelectorAll(".sidebar-tree .toctree-l1");
    sidebarLinks.forEach((li) => {
      const a = li.querySelector("a");
      if (!a) return;
      const href = a.getAttribute("href") || "";
      const isJaLink = href.endsWith(".ja.html") || href.includes(".ja.html");
      const isGuideLink = href.includes("guide/") || href.endsWith(".ja.html") ||
        /^(quickstart|plotting|animation|inp|units|boundaries|distributed)/.test(href);

      if (!isGuideLink) return; // don't touch API Reference etc.

      // href="#" means the current page's self-link in the sidebar.
      if (href === "#") {
        if (activeLang === "ja") {
          li.style.display = (isGuideJa || isIndexJa) ? "" : "none";
        } else {
          li.style.display = (isGuideJa || isIndexJa) ? "none" : "";
        }
        return;
      }

      if (activeLang === "ja") {
        li.style.display = isJaLink ? "" : "none";
      } else {
        li.style.display = isJaLink ? "none" : "";
      }
    });
  }

  // --- Dropdown creation ---
  function createSwitcher(container) {
    const select = document.createElement("select");
    select.className = "lang-switcher";
    select.setAttribute("aria-label", "Language");

    const optEn = document.createElement("option");
    optEn.textContent = "English";
    optEn.value = "en";
    optEn.selected = lang === "en";

    const optJa = document.createElement("option");
    optJa.textContent = "日本語";
    optJa.value = "ja";
    optJa.selected = lang === "ja";

    select.appendChild(optEn);
    select.appendChild(optJa);

    select.addEventListener("change", () => {
      const newLang = select.value;
      localStorage.setItem(STORAGE_KEY, newLang);

      const url = counterpartUrl(newLang);
      if (url) {
        window.location.href = url;
      } else {
        filterSidebar(newLang);
        document.querySelectorAll(".lang-switcher").forEach((s) => {
          s.value = newLang;
        });
      }
    });

    container.appendChild(select);
  }

  // --- Inject switchers ---
  const headerToggle = document.querySelector(
    ".theme-toggle-container.theme-toggle-header"
  );
  if (headerToggle && headerToggle.parentNode) {
    const wrapper = document.createElement("div");
    wrapper.className = "lang-switcher-container";
    headerToggle.parentNode.insertBefore(wrapper, headerToggle);
    createSwitcher(wrapper);
  }

  const contentToggle = document.querySelector(
    ".theme-toggle-container.theme-toggle-content"
  );
  if (contentToggle && contentToggle.parentNode) {
    const wrapper = document.createElement("div");
    wrapper.className = "lang-switcher-container";
    contentToggle.parentNode.insertBefore(wrapper, contentToggle);
    createSwitcher(wrapper);
  }

  filterSidebar(lang);
});
