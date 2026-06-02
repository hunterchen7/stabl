# Deploying stabl-server to the Olares box

## 1. Rsync the repo

From your laptop, in `~/Documents/GitHub/stabl`:

```bash
rsync -av --delete \
  --exclude='.git' --exclude='.venv*' \
  --exclude='*.pt' --exclude='examples' \
  ./ olares:~/stabl/
```

(`.pt` excluded — the YOLO weights are 80 MB and not needed for the
KLT/DLC paths the server exposes.)

## 2. Install the API deps in the existing DLC venv

We piggyback on `~/.venv-dlc` so torch/DLC are already available for the
DLC job kinds we'll add next.

```bash
ssh olares '~/.venv-dlc/bin/pip install -e ~/stabl[api]'
```

## 3. Generate + plant the API key

On your **Mac**:

```bash
# Generate the plaintext key. KEEP THIS — paste into keychain below.
KEY=$(openssl rand -hex 32)
echo "$KEY"

# Hash it for the server side.
HASH=$(echo -n "$KEY" | shasum -a 256 | awk '{print $1}')
echo "$HASH"

# Store plaintext in macOS keychain so the local CLI can use it.
security add-generic-password -a stabl -s stabl-api-key -w "$KEY" -U
```

On the **Olares box**:

```bash
ssh olares "cp ~/stabl/deploy/stabl.env.example ~/stabl/deploy/stabl.env"
ssh olares "sed -i 's|^STABL_API_KEY_HASH=.*|STABL_API_KEY_HASH=$HASH|' ~/stabl/deploy/stabl.env"
ssh olares "chmod 600 ~/stabl/deploy/stabl.env"
```

## 4. Install + start the systemd unit

```bash
ssh olares 'sudo cp ~/stabl/deploy/stabl-server.service /etc/systemd/system/'
ssh olares 'sudo systemctl daemon-reload && sudo systemctl enable --now stabl-server'
ssh olares 'sudo systemctl status stabl-server --no-pager'
```

Logs: `ssh olares 'journalctl -u stabl-server -f --no-pager'`

## 5. Cloudflare tunnel

See [CLOUDFLARE.md](./CLOUDFLARE.md).

## 6. Smoke test from Mac

```bash
KEY=$(security find-generic-password -a stabl -s stabl-api-key -w)
curl https://stabl.hunterchen.ca/health
# {"ok":true}
curl -H "Authorization: Bearer $KEY" https://stabl.hunterchen.ca/v1/jobs
# {"jobs":[]}
```
