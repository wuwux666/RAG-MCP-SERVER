<script setup lang="ts">
import { ref, nextTick } from 'vue'
import { query, agenticQuery } from '../api'
import ChatMessage from '../components/ChatMessage.vue'
import QueryInput from '../components/QueryInput.vue'

interface Message {
  role: 'user' | 'assistant'
  content: string
  citations?: any[]
  trace?: any[]
  mode?: string
  loading?: boolean
}

const messages = ref<Message[]>([])
const agenticMode = ref(false)
const loading = ref(false)
const chatContainer = ref<HTMLElement>()

async function sendQuery(text: string) {
  messages.value.push({ role: 'user', content: text })
  messages.value.push({ role: 'assistant', content: '', loading: true })
  loading.value = true

  await nextTick()
  scrollToBottom()

  try {
    const fn = agenticMode.value ? agenticQuery : query
    const result = await fn({ query: text })
    messages.value[messages.value.length - 1] = {
      role: 'assistant',
      content: result.answer,
      citations: result.citations,
      trace: result.trace,
      mode: result.mode,
    }
  } catch (e: any) {
    messages.value[messages.value.length - 1] = {
      role: 'assistant',
      content: `❌ 出错了: ${e.message || '请检查后端是否已启动'}`,
    }
  } finally {
    loading.value = false
    await nextTick()
    scrollToBottom()
  }
}

function scrollToBottom() {
  chatContainer.value?.scrollTo({ top: chatContainer.value.scrollHeight, behavior: 'smooth' })
}
</script>

<template>
  <div class="flex flex-col h-[calc(100vh-5rem)]">
    <!-- Chat message area -->
    <div
      ref="chatContainer"
      class="flex-1 overflow-y-auto px-2 py-4 space-y-2"
    >
      <!-- Empty state -->
      <div
        v-if="messages.length === 0"
        class="flex flex-col items-center justify-center h-full text-center"
      >
        <div class="text-6xl mb-4">🌸</div>
        <h2 class="text-xl font-bold text-sakura-pink mb-2">Sakura RAG</h2>
        <p class="text-sakura-muted">问我任何问题，我会在知识库中为你找到答案~</p>
        <p class="text-sakura-muted text-sm mt-1">开启 Agentic 模式可以处理复杂多跳问题哦 ✨</p>
      </div>

      <!-- Messages -->
      <ChatMessage v-for="(msg, i) in messages" :key="i" :msg="msg" />
    </div>

    <!-- Input area (sticky bottom) -->
    <QueryInput
      @send="sendQuery"
      v-model:agentic="agenticMode"
      :disabled="loading"
    />
  </div>
</template>
