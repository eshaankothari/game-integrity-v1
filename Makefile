# One place for every setup and run command, Python and Node alike.
#
#   make setup     install everything (pip + npm)
#   make demo      run the whole thing  ->  http://localhost:5173
#
# Run `make` on its own to list the targets.

.DEFAULT_GOAL := help
.PHONY: help setup setup-py setup-ui api ui demo duckdb dump restore check clean

PY   ?= python3
DB   ?= game_integrity.duckdb
PGDB ?= game_integrity_v1

help:  ## show this list
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk -F':.*?## ' '{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  First time:   make setup && make demo"
	@echo "  No Postgres?  make setup && GI_DB=$(DB) make demo"

# --- install ---------------------------------------------------------------

setup: setup-py setup-ui  ## install Python AND Node dependencies
	@echo ""
	@echo "Ready. Next:  make demo"

setup-py:  ## pip install -r requirements.txt
	$(PY) -m pip install -r requirements.txt

setup-ui:  ## npm ci in frontend/ (exact lockfile, not a fresh resolve)
	cd frontend && npm ci

# --- run -------------------------------------------------------------------

api:  ## API only, port 8000
	$(PY) -m uvicorn server.app:app --reload --port 8000

ui:  ## dashboard only, port 5173 (proxies /api to 8000)
	cd frontend && npm run dev

demo:  ## API + dashboard together; Ctrl-C stops both
	@echo "API      -> http://localhost:8000/api/docs"
	@echo "dashboard-> http://localhost:5173"
	@echo ""
	@$(PY) -m uvicorn server.app:app --port 8000 & \
	 cd frontend && npm run dev; \
	 kill %1 2>/dev/null || true

# --- database --------------------------------------------------------------

duckdb:  ## export Postgres -> a single .duckdb file for handoff
	$(PY) to_duckdb.py run

dump:  ## export Postgres -> game_integrity.dump (all 21 tables + constraints)
	pg_dump $(PGDB) -Fc -f game_integrity.dump
	@ls -lh game_integrity.dump | awk '{print "  wrote game_integrity.dump", $$5}'

restore:  ## load game_integrity.dump into a FRESH Postgres database
	@# Refuse a non-empty target. pg_restore has no "only if absent" mode, so restoring
	@# over an existing schema emits one error per table, index and constraint -- 114 of
	@# them here -- while the primary keys silently reject the duplicate rows. It looks
	@# catastrophic and changes nothing, which is the worst combination to hand someone.
	@if psql -lqt 2>/dev/null | cut -d'|' -f1 | grep -qw $(PGDB); then \
	  n=$$(psql -d $(PGDB) -tAc "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'"); \
	  if [ "$$n" != "0" ]; then \
	    echo "!! database '$(PGDB)' already exists and holds $$n tables."; \
	    echo "   restore is for an EMPTY database. Either:"; \
	    echo "     PGDB=game_integrity_copy make restore    # somewhere else"; \
	    echo "     dropdb $(PGDB) && make restore           # replace it (destructive)"; \
	    exit 1; \
	  fi; \
	else createdb $(PGDB); fi
	pg_restore -d $(PGDB) --no-owner --no-privileges game_integrity.dump
	@echo "  restored -> $(PGDB).  Run: make demo"

check:  ## confirm the database is present and scored
	@$(PY) -c "import config, db; \
	print('backend :', config.BACKEND); \
	print('shortlist:', db.rows('select count(*) n from player_game_scores where in_shortlist', one=True)['n'], '(expect 3207)')"

clean:  ## remove build output (never touches the database or caches)
	rm -rf frontend/dist frontend/.vite
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
