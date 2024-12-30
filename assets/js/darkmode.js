function toggleDarkMode() {
    const DARK_CLASS = 'dark';

    var body = document.querySelector("body");
    if (body.classList.contains(DARK_CLASS)) {
        setCookie('theme', 'light');
        body.classList.remove(DARK_CLASS);
    } else {
        setCookie('theme', 'dark');
        body.classList.add(DARK_CLASS);
    }
}

function getCookie(name) {
    var v = document.cookie.match('(^|;) ?' + name + '=([^;]*)(;|$)');
    return v ? v[2] : null;
}

function setCookie(name, value, days) {
    var d = new Date();
    d.setTime(d.getTime() + 24 * 60 * 60 * 1000 * days);
    document.cookie = name + "=" + value + ";path=/;SameSite=strict;expires=" + d.toGMTString();
}

function deleteCookie(name) { setCookie(name, '', -1); }

const userPrefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
var theme = getCookie('theme');

// Set dark mode as the default if no cookie is set
if (theme === null) {
    theme = userPrefersDark ? 'dark' : 'dark'; // Default to dark
    setCookie('theme', theme, 365); // Save the default theme to a cookie
}

// Apply the theme based on the cookie value
if (theme === 'dark') {
    document.body.classList.add('dark');
    document.querySelectorAll('.dark-mode-toggle').forEach(ti => ti.checked = true);
} else {
    document.body.classList.remove('dark');
    document.querySelectorAll('.dark-mode-toggle').forEach(ti => ti.checked = false);
}
