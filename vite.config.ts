import { defineConfig } from 'vite'

export default defineConfig({
    esbuild: {
        keepNames: true
    },
    build: {
        target: 'esnext'
    },
    base: process.env.GITHUB_ACTIONS_BASE || undefined
})
