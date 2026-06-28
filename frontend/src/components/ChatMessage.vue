<script setup lang="ts">
import { computed } from 'vue'
import CitationCard from './CitationCard.vue'
import AgenticTrace from './AgenticTrace.vue'

interface Message {
  role: 'user' | 'assistant'
  content: string
  citations?: any[]
  trace?: any[]
  mode?: string
  loading?: boolean
}

const props = defineProps<{ msg: Message }>()

function renderMarkdown(text: string): string {
  let html = text
    // bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // italic
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // inline code
    .replace(/`(.+?)`/g, '<code class="bg-sakura-light/30 px-1 rounded text-xs">$1</code>')
    // newlines to <br>
    .replace(/\n/g, '<br>')

  return html
}

const renderedContent = computed(() => {
  if (!props.msg.content) return ''
  return renderMarkdown(props.msg.content)
})
</script>

<template>
  <div
    class="flex gap-3 mb-4"
    :class="msg.role === 'user' ? 'flex-row-reverse' : ''"
  >
    <!-- Avatar -->
    <div
      class="w-8 h-8 rounded-full flex items-center justify-center text-lg shrink-0"
      :class="msg.role === 'user' ? 'bg-sakura-light' : 'bg-sakura-pink'"
    >
      {{ msg.role === 'user' ? '👤' : '🤖' }}
    </div>

    <!-- Bubble -->
    <div
      class="max-w-[75%] rounded-2xl px-4 py-3 shadow-sakura"
      :class="msg.role === 'user'
        ? 'bg-white text-sakura-text rounded-tr-md'
        : 'bg-sakura-pink/10 text-sakura-text rounded-tl-md'"
    >
      <!-- Loading dots -->
      <div v-if="msg.loading" class="flex gap-1 py-2">
        <span class="w-2 h-2 bg-sakura-pink rounded-full animate-bounce" style="animation-delay: 0s"></span>
        <span class="w-2 h-2 bg-sakura-pink rounded-full animate-bounce" style="animation-delay: 0.15s"></span>
        <span class="w-2 h-2 bg-sakura-pink rounded-full animate-bounce" style="animation-delay: 0.3s"></span>
      </div>

      <!-- Content -->
      <div v-else class="text-sm leading-relaxed" v-html="renderedContent"></div>

      <!-- Agentic mode badge -->
      <div v-if="msg.mode === 'agentic' && !msg.loading" class="mt-1">
        <span class="text-xs text-sakura-pink/70 font-medium">🌸 Agentic · {{ msg.trace?.length || 0 }} 轮推理</span>
      </div>

      <!-- Citations -->
      <CitationCard v-if="msg.citations?.length" :citations="msg.citations" />

      <!-- Agentic trace -->
      <AgenticTrace v-if="msg.trace?.length" :trace="msg.trace" />
    </div>
  </div>
</template>
