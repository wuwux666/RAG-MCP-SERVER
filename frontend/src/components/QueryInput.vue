<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  disabled: boolean
  agentic: boolean
}>()

const emit = defineEmits<{
  send: [text: string]
  'update:agentic': [val: boolean]
}>()

const input = ref('')

function send() {
  const text = input.value.trim()
  if (!text || props.disabled) return
  emit('send', text)
  input.value = ''
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    send()
  }
}
</script>

<template>
  <div class="border-t border-sakura-light/30 bg-white/50 backdrop-blur-sm p-4">
    <div class="max-w-3xl mx-auto">
      <!-- Agentic toggle -->
      <div class="flex items-center gap-2 mb-2">
        <label class="flex items-center gap-1.5 cursor-pointer select-none">
          <input
            type="checkbox"
            :checked="agentic"
            @change="emit('update:agentic', ($event.target as HTMLInputElement).checked)"
            class="accent-sakura-pink w-4 h-4"
          />
          <span class="text-xs text-sakura-muted">🌸 Agentic 模式</span>
        </label>
        <span class="text-xs text-sakura-muted/60">| 多步推理，适合复杂问题</span>
      </div>

      <!-- Input row -->
      <div class="flex gap-2">
        <input
          v-model="input"
          @keydown="onKeydown"
          :disabled="disabled"
          placeholder="输入你的问题..."
          class="flex-1 px-4 py-2.5 rounded-2xl border border-sakura-light/50 bg-white
                 text-sakura-text placeholder-sakura-muted/50
                 focus:outline-none focus:border-sakura-pink focus:ring-2 focus:ring-sakura-light/30
                 transition-all disabled:opacity-50"
        />
        <button
          @click="send"
          :disabled="disabled || !input.trim()"
          class="px-5 py-2.5 rounded-2xl font-medium text-white
                 bg-gradient-to-r from-sakura-pink to-sakura-light
                 hover:from-sakura-accent hover:to-sakura-pink
                 disabled:opacity-50 disabled:cursor-not-allowed
                 transition-all duration-200 shadow-sakura hover:shadow-lg
                 active:scale-95"
        >
          🌸 发送
        </button>
      </div>
    </div>
  </div>
</template>
