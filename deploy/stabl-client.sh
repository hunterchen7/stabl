#!/usr/bin/env bash
# Tiny CLI wrapper for the stabl-server API.
# Pulls the bearer token from macOS keychain on demand.
# Install:  cp deploy/stabl-client.sh ~/bin/stabl && chmod +x ~/bin/stabl
set -euo pipefail

BASE="${STABL_BASE:-https://stabl.hunterchen.ca}"
KEY="$(security find-generic-password -a stabl -s stabl-api-key -w 2>/dev/null)" || {
  echo "no API key in keychain; see deploy/SETUP.md step 3" >&2; exit 1; }
AUTH=(-H "Authorization: Bearer $KEY")

cmd="${1:-help}"; shift || true
case "$cmd" in
  health)  curl -fsS "$BASE/health"; echo ;;
  upload)  # stabl upload <path> [csv|clip]  — multipart, 100MB cap on free CF
    path="$1"; kind="${2:-clip}"
    curl -fsS "${AUTH[@]}" -F "kind=$kind" -F "file=@$path" "$BASE/v1/upload" ;;
  upload-rsync) # stabl upload-rsync <path> [csv|clip]  — for big files via Tailscale
    path="$1"; kind="${2:-clip}"
    reserve=$(curl -fsS "${AUTH[@]}" -H "Content-Type: application/json" \
      -d "{\"kind\":\"$kind\"}" "$BASE/v1/upload/reserve")
    file_id=$(printf '%s' "$reserve" | sed -n 's/.*"file_id":"\([^"]*\)".*/\1/p')
    target=$(printf '%s' "$reserve" | sed -n 's/.*"rsync_target":"\([^"]*\)".*/\1/p')
    [ -z "$file_id" ] && { echo "reserve failed: $reserve" >&2; exit 1; }
    echo "rsync to olares:$target ..." >&2
    rsync -av "$path" "olares:$target" >&2
    curl -fsS "${AUTH[@]}" "$BASE/v1/files/$file_id/info" ;;
  upload-r2) # stabl upload-r2 <path> [csv|clip]  — direct PUT to R2, any size
    path="$1"; kind="${2:-clip}"
    presign=$(curl -fsS "${AUTH[@]}" -H "Content-Type: application/json" \
      -d "{\"kind\":\"$kind\"}" "$BASE/v1/upload/r2/presign")
    file_id=$(printf '%s' "$presign" | sed -n 's/.*"file_id":"\([^"]*\)".*/\1/p')
    r2_key=$(printf '%s' "$presign" | sed -n 's/.*"r2_key":"\([^"]*\)".*/\1/p')
    url=$(printf '%s' "$presign" | python3 -c 'import sys,json; print(json.load(sys.stdin)["presigned_url"])')
    [ -z "$file_id" ] && { echo "presign failed: $presign" >&2; exit 1; }
    echo "PUT $(basename "$url") (presigned)..." >&2
    curl -fsS -X PUT -T "$path" -H "Content-Type: application/octet-stream" "$url" >/dev/null
    curl -fsS "${AUTH[@]}" -H "Content-Type: application/json" \
      -d "{\"file_id\":\"$file_id\",\"r2_key\":\"$r2_key\"}" "$BASE/v1/upload/r2/finalize" ;;
  job)     # stabl job <kind> '<json-params>'
    kind="$1"; params="$2"
    curl -fsS "${AUTH[@]}" -H "Content-Type: application/json" \
      -d "$params" "$BASE/v1/jobs/$kind" ;;
  status)  # stabl status <job_id>
    curl -fsS "${AUTH[@]}" "$BASE/v1/jobs/$1" ;;
  jobs)    curl -fsS "${AUTH[@]}" "$BASE/v1/jobs" ;;
  get)     # stabl get <file_id> <out-path>
    curl -fsS "${AUTH[@]}" -o "$2" "$BASE/v1/files/$1" ;;
  restart) curl -fsS "${AUTH[@]}" -X POST "$BASE/v1/restart" ;;
  version) curl -fsS "$BASE/v1/version" ;;
  sync)    # stabl sync — pulls origin/main on the server, then restarts
    curl -fsS "${AUTH[@]}" -X POST "$BASE/v1/sync" ;;
  check)   # quick local vs remote parity check
    local_sha=$(cd ~/Documents/GitHub/stabl && git rev-parse HEAD 2>/dev/null)
    remote=$(curl -fsS "$BASE/v1/version")
    remote_sha=$(printf '%s' "$remote" | sed -n 's/.*"sha":"\([^"]*\)".*/\1/p')
    echo "local : $local_sha"
    echo "remote: $remote_sha"
    [ "$local_sha" = "$remote_sha" ] && echo "IN SYNC" || echo "DIVERGED — push then run: stabl sync"
    ;;
  *)       echo "usage: stabl {health|upload|upload-rsync|upload-r2|job|status|jobs|get|deploy|restart} ..." >&2; exit 2 ;;
esac
