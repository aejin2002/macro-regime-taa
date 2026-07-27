# External CSV inputs

These files are optional. Their absence only disables the specific model(s)
that depend on them — see `docs/data_dictionary.md` for exact schemas.

| File | Enables |
|---|---|
| `conference_board_lei.csv` | Growth Model A (Conference Board LEI variant, if used instead of OECD CLI) |
| `ism_new_orders.csv` | Growth Model C |
| `ism_prices_paid.csv` | Inflation Model B (full, ISM-inclusive version) |
| `cleveland_fed_inflation_nowcast.csv` | Inflation Model C (disabled by default, see methodology) |

None of these files are committed to the repository (see `.gitignore` policy
in the project README) — supply your own locally.
