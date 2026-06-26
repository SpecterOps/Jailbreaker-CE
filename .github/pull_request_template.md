## Summary

## Testing

- [ ] `uv run --locked --extra dev --extra docs --extra postgres python -m pytest backend/tests`
- [ ] `node --test frontend/tests/*.test.mjs`
- [ ] `docker compose -f docker-compose.yml config --quiet`

## Checklist

- [ ] I did not commit secrets, `.env`, logs, or private datasets.
- [ ] I updated docs or tests when behavior changed.
- [ ] I reviewed `SECURITY.md` before including sensitive content.
