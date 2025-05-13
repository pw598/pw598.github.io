(function() {
  const userPref = localStorage.getItem("theme");
  const systemPrefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;

  // Default to dark mode if no user preference
  if (!userPref && systemPrefersDark) {
    document.documentElement.setAttribute("data-theme", "dark");
  } else if (!userPref) {
    // Force dark mode even if system prefers light
    document.documentElement.setAttribute("data-theme", "dark");
  } else {
    document.documentElement.setAttribute("data-theme", userPref);
  }

  document.addEventListener("DOMContentLoaded", function() {
    const toggleBtn = document.getElementById("theme-toggle");
    if (!toggleBtn) return;

    toggleBtn.addEventListener("click", function() {
      const currentTheme = document.documentElement.getAttribute("data-theme");
      const newTheme = currentTheme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", newTheme);
      localStorage.setItem("theme", newTheme);
    });
  });
})();
