<script setup lang="ts">
import { computed } from 'vue'

interface Stage {
  name: string
  elapsed_ms: number
}

const props = defineProps<{
  stages: Stage[]
}>()

const maxMs = computed(() => {
  if (!props.stages?.length) return 1
  return Math.max(...props.stages.map(s => s.elapsed_ms), 1)
})

const totalMs = computed(() => {
  if (!props.stages?.length) return 0
  return props.stages.reduce((sum, s) => sum + s.elapsed_ms, 0)
})

function pct(ms: number): string {
  return ((ms / maxMs.value) * 100).toFixed(1) + '%'
}
</script>

<template>
  <div class="space-y-2">
    <div class="flex items-center gap-2 text-xs text-sakura-muted mb-2">
      <span>总耗时: {{ totalMs }}ms</span>
    </div>
    <div
      v-for="stage in stages"
      :key="stage.name"
      class="flex items-center gap-3"
    >
      <!-- Stage label -->
      <span class="text-xs text-sakura-text w-20 shrink-0 text-right font-medium">
        {{ stage.name }}
      </span>
      <!-- Bar -->
      <div class="flex-1 h-6 bg-sakura-light/20 rounded-lg overflow-hidden relative">
        <div
          class="h-full rounded-lg bg-gradient-to-r from-sakura-pink to-sakura-light flex items-center justify-end pr-2 transition-all duration-500"
          :style="{ width: pct(stage.elapsed_ms) }"
        >
          <span class="text-xs text-white font-semibold drop-shadow-sm">
            {{ stage.elapsed_ms }}ms
          </span>
        </div>
      </div>
    </div>
  </div>
</template>
