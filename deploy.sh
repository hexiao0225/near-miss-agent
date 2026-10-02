#!/usr/bin/env bash
# Deploy Near-Miss Agent to this team's namespace at http://<team host>/app.
# Run on the workshop VM from this directory: bash deploy.sh
# Follows .cursor/skills/deployment/deploy-app-no-registry: public python image,
# code from a ConfigMap, credentials from a Secret, Ingress path /app.
set -euo pipefail
cd "$(dirname "$0")"

export KUBECONFIG=/config/kubeconfig
mapfile -t TEAM_CONFIGS < <(find /config -maxdepth 1 -type f -name '*.config' | sort)
(( ${#TEAM_CONFIGS[@]} == 1 )) || { echo "expected exactly one /config/*.config"; exit 1; }
set -a && source "${TEAM_CONFIGS[0]}" && set +a

NS="$USERNAME"
APP_NAME=near-miss-agent
APP_PORT=8080
APP_HOST="${INGRESS_URL#http://}"; APP_HOST="${APP_HOST#https://}"; APP_HOST="${APP_HOST%%/*}"
WANDB_MODEL="${WANDB_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

kubectl -n "$NS" create configmap "${APP_NAME}-code" \
  --from-file=main.py --from-file=index.html --from-file=fixtures_hits.json \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$NS" create secret generic "${APP_NAME}-creds" \
  --from-literal=VSS_URL="$INGRESS_URL" \
  --from-literal=VSS_USERNAME="$USERNAME" \
  --from-literal=VSS_PASSWORD="$PASSWORD" \
  --from-literal=WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --from-literal=WANDB_TEAM="${WANDB_TEAM:-}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$NS" apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${APP_NAME}
  labels: {app: ${APP_NAME}}
spec:
  replicas: 1
  selector: {matchLabels: {app: ${APP_NAME}}}
  template:
    metadata:
      labels: {app: ${APP_NAME}}
    spec:
      containers:
      - name: app
        image: python:3.12-slim
        imagePullPolicy: IfNotPresent
        ports: [{containerPort: ${APP_PORT}}]
        env:
        - {name: PORT, value: "${APP_PORT}"}
        - {name: WANDB_MODEL, value: "${WANDB_MODEL}"}
        envFrom:
        - secretRef: {name: ${APP_NAME}-creds}
        volumeMounts: [{name: code, mountPath: /code}]
        workingDir: /code
        command: ["python", "main.py"]
        readinessProbe:
          httpGet: {path: /health, port: ${APP_PORT}}
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: code
        configMap: {name: ${APP_NAME}-code}
---
apiVersion: v1
kind: Service
metadata:
  name: ${APP_NAME}
  labels: {app: ${APP_NAME}}
spec:
  selector: {app: ${APP_NAME}}
  ports: [{name: http, port: 80, targetPort: ${APP_PORT}}]
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ${APP_NAME}
  labels: {app: ${APP_NAME}}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /\$2
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
spec:
  ingressClassName: nginx
  rules:
  - host: ${APP_HOST}
    http:
      paths:
      - path: /app(/|$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: ${APP_NAME}
            port: {number: 80}
EOF

kubectl -n "$NS" rollout restart deploy/"$APP_NAME"
kubectl -n "$NS" rollout status deploy/"$APP_NAME" --timeout=120s
curl -sS "http://${APP_HOST}/app/health"; echo
echo "Live: http://${APP_HOST}/app   (demo shortcut: http://${APP_HOST}/app#scan)"
