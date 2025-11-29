# algebra_ops

<!--toc:start-->

- [algebra_ops](#algebraops)
  - [Why are we doing this?](#why-are-we-doing-this)
  - [Who is this for?](#who-is-this-for)
  - [Technical Details](#technical-details)
  - [Author](#author)
  <!--toc:end-->

**My implementation of algebraic operations made in Rust.**

This project is a personal exploration into the low-level math needed to build neural networks from scratch.

## Why are we doing this?

I created this project to learn. I want to understand exactly how things work under the hood. My goals are:

- **To learn Rust:** Understanding lifetimes, generics, and memory management.
- **To learn Math:** deeply understanding matrix operations, Tensors, and linear algebra.
- **To build a foundation:** This is meant to be a starting base to eventually build neural networks without relying on "black box" libraries.

## Who is this for?

**This is for me and me alone.**

This code is **not** production-ready. It is an educational experiment. It likely contains bugs and is not optimized for speed like professional libraries (e.g., `ndarray` or `burn`). If you are looking for a stable library for your company or app, please look elsewhere.

---

## Technical Details

- **Language:** Rust (Edition 2024)
- **Dependencies:** Uses `num-traits` to handle generic numbers.
- **Quality Control:** Configured with `clippy::all = "warn"` to ensure clean code habits.
- **License:** LGPL 3.0 or later

## Author

Roberto Di Rosa
