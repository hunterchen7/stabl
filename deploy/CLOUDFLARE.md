# Cloudflare Tunnel setup (stabl.hunterchen.ca)

The server runs on `localhost:8000` on Olares; cloudflared exposes it at
`https://stabl.hunterchen.ca` without opening any port on the host. We
use the **dashboard-managed** tunnel pattern — ingress lives in the CF
dashboard, not in a local YAML — so the host only needs the tunnel token.

## 1. Create the tunnel in the dashboard

1. dash.cloudflare.com → Zero Trust → Networks → Tunnels → **Create a tunnel**
2. Connector type: **cloudflared**
3. Name: `stabl`
4. Copy the token shown on the next page — that's the long base64-ish
   string starting with `eyJ...`. Save it; you paste it once on Olares.
5. Under "Public hostname", add:

   | Subdomain | Domain          | Service                |
   |-----------|-----------------|------------------------|
   | `stabl`   | `hunterchen.ca` | `HTTP` `localhost:8000`|

   Cloudflare will create the `CNAME stabl.hunterchen.ca → <uuid>.cfargotunnel.com`
   record automatically.

## 2. Install cloudflared on Olares

```bash
ssh olares <<'SH'
curl -L --output /tmp/cloudflared.deb \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i /tmp/cloudflared.deb
rm /tmp/cloudflared.deb
SH
```

## 3. Install the tunnel as a systemd service

The `service install` subcommand drops a systemd unit that runs
`cloudflared tunnel run --token <token>` and survives reboots:

```bash
ssh olares 'sudo cloudflared service install <TOKEN-FROM-STEP-1>'
ssh olares 'sudo systemctl status cloudflared'
# expect Active: active (running)
```

To rotate the token later: dashboard → tunnel → Configure → Rotate token,
then re-run `sudo cloudflared service install <NEW>`.

## 4. Verify

Once `stabl-server.service` is also running (see SETUP.md):

```bash
curl https://stabl.hunterchen.ca/health
# → {"ok":true}
```

If you see Cloudflare's "1033 / Argo Tunnel error" page, cloudflared isn't
running or the ingress in the dashboard isn't pointing at port 8000.
