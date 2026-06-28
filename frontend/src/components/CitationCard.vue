<script setup lang="ts">
defineProps<{ citations: Array<{ index: number; source: string; score: number }> }>()
</script>

<template>
  <div class="mt-2 pt-2 border-t border-sakura-light/50">
    <div class="text-xs text-sakura-muted mb-1 font-medium">📎 引用来源</div>
    <div class="flex flex-wrap gap-1.5">
      <span
        v-for="c in citations"
        :key="c.index"
        class="inline-flex items-center gap-1 px-2 py-0.5 rounded-lg text-xs
               bg-sakura-light/20 text-sakura-text hover:bg-sakura-light/40 transition-colors cursor-default"
      >
        [{{ c.index }}] {{ c.source }}
      </span>
    </div>
  </div>
</template>
