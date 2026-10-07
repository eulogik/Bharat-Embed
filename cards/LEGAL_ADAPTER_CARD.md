# Bharat legal adapter card draft (adapter only)

Same base as Bharat-Embed 270M plus legal LoRA. Adapter only repo. Small download like 14MB pattern.

## Use

- DIFC, ADGM, GST, VAT statute retrieval. EN, AR, HI queries.
- Swap adapter on same base. No extra base weights.
- Train: 1 epoch on legal_10k only.

## Gate

- Statute recall at 5 plus 0.10 over generic on held out DIFC and GST queries.
- Human spot check before publish. Synthetic queries alone are not proof.

## License

Apache-2.0 for our synthetic slice. Cite Banking and MARCO sources where reused.
