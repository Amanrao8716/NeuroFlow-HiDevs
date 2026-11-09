FROM node:20-slim AS builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ARG NEXT_PUBLIC_API_URL=/api
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
RUN npm run build

# The UI is client-rendered and has no Next.js server-only routes, so export
# static files and serve them from Nginx. Nginx also proxies API and SSE calls.
FROM nginx:alpine AS runtime
COPY --from=builder /app/frontend/out/ /usr/share/nginx/html/
COPY --from=builder /app/frontend/.next/static/ /usr/share/nginx/html/_next/static/
COPY --from=builder /app/frontend/public/ /usr/share/nginx/html/
COPY infra/nginx/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q --spider http://127.0.0.1/ || exit 1
