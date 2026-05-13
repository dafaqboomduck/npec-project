# =========================================
# Stage 1: Build the React (Vite) application
# =========================================
FROM node:20-alpine AS builder
WORKDIR /app

# Copy lockfiles first — Docker caches this layer and only
# reruns npm ci when the lockfile actually changes
COPY package.json package-lock.json* ./

# Clean, reproducible install based on the lockfile
RUN --mount=type=cache,target=/root/.npm npm ci

# VITE_API_URL must be injected at BUILD TIME, not runtime.
# Vite bakes all VITE_* variables into the compiled JS bundle,
# so they must be available during `npm run build`.
# Pass this with: docker build --build-arg VITE_API_URL=https://your-api.com
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

# Copy the rest of the application source code
COPY ./ ./

# Compile the React app — output goes to /app/dist
RUN npm run build

# =========================================
# Stage 2: Serve the production build
# =========================================
# Start from a fresh, minimal image — no build tools, no source code
FROM node:20-alpine AS runner

# Run Node in production mode
ENV NODE_ENV=production
WORKDIR /app

# Copy only the compiled static files from the builder stage
COPY --from=builder /app/dist ./dist

RUN npm install -g serve@14

# Drop root privileges for security best practices
USER node

# Expose the port used by `serve`
EXPOSE 3000

# Serve
CMD ["serve", "-s", "dist", "-l", "3000"]