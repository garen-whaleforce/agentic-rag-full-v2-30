#!/usr/bin/env sh
set -e

RAG_PATH=${EARNINGS_RAG_PATH:-/app/EarningsCallAgenticRag}
RAG_REPO_URL=${EARNINGS_RAG_REPO_URL:-https://github.com/la9806958/EarningsCallAgenticRag.git}
CRED_PATH="$RAG_PATH/credentials.json"

# Ensure external repo is available (clone if missing)
if [ ! -d "$RAG_PATH/.git" ]; then
  echo "Cloning external repo to $RAG_PATH ..."
  rm -rf "$RAG_PATH"
  git clone --depth 1 "$RAG_REPO_URL" "$RAG_PATH"
else
  echo "External repo already present at $RAG_PATH"
fi

cat > "$CRED_PATH" <<EOF
{
  "openai_api_key": "${OPENAI_API_KEY:-}",
  "neo4j_uri": "${NEO4J_URI:-}",
  "neo4j_username": "${NEO4J_USERNAME:-}",
  "neo4j_password": "${NEO4J_PASSWORD:-}",
  "neo4j_database": "${NEO4J_DATABASE:-neo4j}"
}
EOF

echo "Generated credentials.json at $CRED_PATH"

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
