/**
 * Language switcher for emout docs.
 *
 * Adds a Japanese/English dropdown next to furo's dark-mode toggle.
 * Guide pages use the *.ja.html / *.html naming convention.
 */
document.addEventListener("DOMContentLoaded", () => {
  const path = window.location.pathname;
  const isJa = path.endsWith(".ja.html") || path.endsWith(".ja/");

  // Compute the counterpart URL (only for guide pages)
  const isGuide = path.includes("/guide/");
  let otherUrl;
  if (isGuide) {
    otherUrl = isJa
      ? path.replace(/\.ja\.html$/, ".html")
      : path.replace(/\.html$/, ".ja.html");
  } else {
    otherUrl = null; // no counterpart
  }

  function createSwitcher(container) {
    const select = document.createElement("select");
    select.className = "lang-switcher";
    select.setAttribute("aria-label", "Language");

    const optEn = document.createElement("option");
    optEn.textContent = "English";
    optEn.value = "en";
    optEn.selected = !isJa;

    const optJa = document.createElement("option");
    optJa.textContent = "日本語";
    optJa.value = "ja";
    optJa.selected = isJa;

    select.appendChild(optEn);
    select.appendChild(optJa);

    if (otherUrl) {
      select.addEventListener("change", () => {
        window.location.href = otherUrl;
      });
    } else {
      select.disabled = true;
      select.title = "Language switching is available on guide pages";
    }

    container.appendChild(select);
  }

  // Header (mobile + desktop top bar)
  const headerToggle = document.querySelector(
    ".theme-toggle-container.theme-toggle-header"
  );
  if (headerToggle && headerToggle.parentNode) {
    const wrapper = document.createElement("div");
    wrapper.className = "lang-switcher-container";
    headerToggle.parentNode.insertBefore(wrapper, headerToggle);
    createSwitcher(wrapper);
  }

  // Content area (desktop, inside article header)
  const contentToggle = document.querySelector(
    ".theme-toggle-container.theme-toggle-content"
  );
  if (contentToggle && contentToggle.parentNode) {
    const wrapper = document.createElement("div");
    wrapper.className = "lang-switcher-container";
    contentToggle.parentNode.insertBefore(wrapper, contentToggle);
    createSwitcher(wrapper);
  }
});
