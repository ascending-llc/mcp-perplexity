FROM node:22.12-alpine AS builder

COPY . /app

WORKDIR /app

RUN --mount=type=cache,target=/root/.npm npm install

# Build the TypeScript code
RUN npm run build

FROM node:22-alpine AS release

WORKDIR /app

COPY --from=builder /app/dist /app/dist
COPY --from=builder /app/package.json /app/package.json
COPY --from=builder /app/package-lock.json /app/package-lock.json

ENV NODE_ENV=production
ENV PORT=3000
ENV HTTP_MODE=true
ENV DEBUG=false

# Expose the HTTP port
EXPOSE 3000

RUN npm ci --ignore-scripts --omit-dev

ENTRYPOINT ["node", "dist/index.js"]
