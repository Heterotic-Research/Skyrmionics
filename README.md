# Skyrmion
LLGS-based spin dynamics simulator and hardware synthesizer

## Documentation

- API docs are published via GitHub Pages from `target/doc`.
	- On each push to `main`, the workflow `.github/workflows/docs.yml` builds docs with `cargo doc --no-deps` and deploys to the `gh-pages` branch.
	- After the first successful run, enable Pages in the repo settings: Pages → Branch: `gh-pages`, Folder: `/`.
	- The documentation URL will be visible in the repo settings; set a custom domain by adding a `CNAME` in the publish step.

## Quick start

```rust
use Skyrmion::{Vector3, LLGSParameters, simulate_llgs};

let params = LLGSParameters { gyromagnetic_ratio: 1.0, damping: 0.1 };
let m0 = Vector3::new(1.0, 0.0, 0.0);
let result = simulate_llgs(m0, &params, 1e-3, 1.0, |t, _m| {
		Vector3::new(0.0, 0.0, 1.0)
});
```

## Developing

- Build: `cargo build`
- Test: `cargo test`
- Docs (local): `cargo doc --open`
