<script setup lang="ts">
import { useRoute } from 'vue-router'

const route = useRoute()
const links = [
  { to: '/chat', label: 'Chat', icon: '💬' },
  { to: '/documents', label: '文档', icon: '📁' },
  { to: '/traces', label: '追踪', icon: '🔍' },
  { to: '/evaluation', label: '评估', icon: '📊' },
]
</script>

<template>
  <nav class="bg-white/80 backdrop-blur-sm border-b border-sakura-light/30 sticky top-0 z-50">
    <div class="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
      <router-link to="/chat" class="flex items-center gap-2 text-sakura-pink font-bold text-lg no-underline">
        <span>🌸</span>
        <span>Sakura RAG</span>
      </router-link>
      <div class="flex gap-1">
        <router-link
          v-for="link in links"
          :key="link.to"
          :to="link.to"
          class="px-3 py-1.5 rounded-xl text-sm transition-all duration-200 no-underline"
          :class="route.path === link.to
            ? 'bg-sakura-pink text-white'
            : 'text-sakura-muted hover:bg-sakura-light/20 hover:text-sakura-text'"
        >
          {{ link.icon }} {{ link.label }}
        </router-link>
      </div>
    </div>
  </nav>
</template>
