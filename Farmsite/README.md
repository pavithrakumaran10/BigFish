# Orchard & Co. — Organic Fruit Store 🍏

A single-page, glassmorphism storefront for selling organic fruit online. Pure HTML/CSS/JS — no build step, no backend. Perfect for GitHub Pages.

## Features
- **Glassmorphism UI** — frosted-glass cards, drawers and nav (`backdrop-filter` blur + saturate), soft gradients, depth shadows, floating orbs.
- **Fruit catalog** — image cards with price, unit, category, stock badge and description. Live search + category filters.
- **Cart + WhatsApp ordering** — add to basket, adjust quantity, then "Order on WhatsApp" opens a pre-filled message to the shop's number.
- **Owner / admin mode** — add or remove fruits and edit shop settings. Changes persist in the browser via `localStorage`.
- **Modern, accessible** — GPU-friendly `transform`/`opacity` transitions, scroll-reveal, `prefers-reduced-motion` fallback, `will-change` hints, attractive Google Fonts (Fraunces + Plus Jakarta Sans).

## How to use as the owner
1. Scroll to the footer and click **Owner login** (or press **Shift + A**).
2. Password: `orchard2026` — **change this** in `index.html` (search for `orchard2026`).
3. Add fruits, mark items sold out, remove fruits, and set your **WhatsApp number** (country code, no `+`, e.g. `91...`).

> Note: localStorage changes are saved in *your* browser only. To change what every visitor sees, edit the `DEFAULT_FRUITS` array and `settings` default in `index.html`, then re-commit.

## Customize
- **Shop name:** edit `Orchard & Co.` in `index.html`.
- **Default WhatsApp number:** change `wa:"919876543210"` near the top of the script.
- **Default fruits:** edit the `DEFAULT_FRUITS` array (name, price, unit, category, emoji, image URL, stock, description).
- **Colors/fonts:** edit the CSS variables in `:root`.

## Hosting on GitHub Pages
1. Push this folder to a GitHub repository.
2. Repo **Settings → Pages → Build and deployment → Source: Deploy from a branch**.
3. Branch: `main`, folder: `/ (root)`. Save.
4. Your site goes live at `https://<username>.github.io/<repo>/` within a minute or two.

## Files
- `index.html` — the entire website (markup, styles, logic).
- `README.md` — this file.
