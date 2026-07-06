# Changelog

## 0.1.0

Initial release.

### Added

- `PMPMProjection` — credibility-blended, trended, plan-adjusted claims PMPM with a
  large-claim pooling load. Blends the group's own PMPM with the book PMPM at a
  supplied or claim-count-derived credibility, trends to the prospective midpoint,
  and applies seasonality onto membership.
- `PremiumRollforward` — rolls a stored premium forward by rate action and plan
  change (level per member-month); not rebuilt from loss experience.
- `GroupProjection` — one group's forward projection: premium roll-forward plus
  credibility-blended claims on the given monthly membership, weighted by a supplied
  `renewal_prob` (e.g. from underwriting). The unit to loop over the book.
- `BookProjection` — aggregates in-force renewals and new business into a book
  budget: expected premium, claims, and loss ratio by group and by month.
- `new_business` — projects a sold case with no experience (`credibility=0`) at the
  manual/book PMPM and a close ratio.
- Frozen `*Result` dataclasses and lowercase functional wrappers (`project_pmpm`,
  `project_group`, `project_book`).
- Built on `actuarialpy` primitives (`credibility_weighted_estimate`,
  `midpoint_trend_factor`, `seasonality_factors`, `pure_premium`); depends only
  downward on `actuarialpy`.
