<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  trace: Array<{ round: number; thought: string; action: string; observation: string }>
}>()

const expanded = ref(false)
</script>

<template>
  <div class="mt-2 pt-2 border-t border-sakura-light/50">
    <button
      @click="expanded = !expanded"
      class="text-xs text-sakura-pink hover:text-sakura-accent transition-colors font-medium"
    >
      {{ expanded ? '▲' : '▼' }} 推理过程 ({{ trace.length }} 轮)
    </button>
    <div v-if="expanded" class="mt-2 space-y-2">
      <div
        v-for="t in trace"
        :key="t.round"
        class="bg-sakura-bg rounded-xl p-2.5 text-xs"
      >
        <div class="font-semibold text-sakura-pink mb-1">
          {{ t.round === trace.length ? '✅' : '🔄' }} Round {{ t.round }}
        </div>
        <div class="text-sakura-muted mb-1">
          <span class="font-medium">Thought:</span> {{ t.thought }}
        </div>
        <div class="text-sakura-muted mb-1">
          <span class="font-medium">Action:</span>
          <code class="bg-sakura-light/30 px-1 rounded">{{ t.action }}</code>
        </div>
        <div class="text-sakura-muted truncate">
          <span class="font-medium">Obs:</span>
          {{ t.observation?.slice(0, 100) }}{{ t.observation?.length > 100 ? '...' : '' }}
        </div>
      </div>
    </div>
  </div>
</template>
