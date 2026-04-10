/**
 * Language switcher for emout docs.
 *
 * - Adds a Japanese/English dropdown next to furo's dark-mode toggle.
 * - Filters the sidebar toctree to show only the active language.
 * - Persists the choice in localStorage.
 *
 * Guide pages use the *.ja.html / *.html naming convention.
 */
document.addEventListener("DOMContentLoaded", () => {
  const STORAGE_KEY = "emout-docs-lang";
  const path = window.location.pathname;
  const isJaPage = path.endsWith(".ja.html") || path.endsWith(".ja/");

  // Determine initial language: page URL > localStorage > default ja
  const lang = isJaPage ? "ja" : (localStorage.getItem(STORAGE_KEY) || "ja");

  // Compute the counterpart URL (only for guide pages)
  const isGuide = path.includes("/guide/");
  function counterpartUrl(targetLang) {
    if (!isGuide) return null;
    if (targetLang === "ja" && !isJaPage) {
      return path.replace(/\.html$/, ".ja.html");
    }
    if (targetLang === "en" && isJaPage) {
      return path.replace(/\.ja\.html$/, ".html");
    }
    return null; // already on the target language
  }

  // --- Sidebar filtering ---
  function filterSidebar(activeLang) {
    const sidebarLinks = document.querySelectorAll(".sidebar-tree .toctree-l1");
    sidebarLinks.forEach((li) => {
      const a = li.querySelector("a");
      if (!a) return;
      const href = a.getAttribute("href") || "";
      // Only filter guide entries (those linking to *.ja.html or guide/*.html)
      const isJaLink = href.endsWith(".ja.html") || href.includes(".ja.html");
      const isGuideLink = href.includes("guide/") || href.endsWith(".ja.html") ||
        // relative links within guide pages
        /^(quickstart|plotting|animation|inp|units|boundaries|distributed)/.test(href);

      if (!isGuideLink) return; // don't touch API Reference etc.

      if (activeLang === "ja") {
        li.style.display = isJaLink || li.classList.contains("current-page") && isJaPage ? "" : "none";
        // Show Japanese entries; for current page (which uses # href), check page lang
        if (href === "#") {
          li.style.display = isJaPage ? "" : "none";
        } else {
          li.style.display = isJaLink ? "" : "none";
        }
      } else {
        if (href === "#") {
          li.style.display = isJaPage ? "none" : "";
        } else {
          li.style.display = isJaLink ? "none" : "";
        }
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

      // Navigate to counterpart if on a guide page
      const url = counterpartUrl(newLang);
      if (url) {
        window.location.href = url;
      } else {
        // Non-guide page: just update sidebar filtering
        filterSidebar(newLang);
        // Sync all switcher dropdowns on the page
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

  // --- Apply initial sidebar filter ---
  filterSidebar(lang);
});
