<script setup lang="ts">
interface MetricCard {
  label: string
  value: string
  icon: string
  description: string
  color: string
}

const metrics: MetricCard[] = [
  {
    label: 'Hit Rate',
    value: '91.2%',
    icon: '🎯',
    description: '检索命中率 — 衡量查询结果中相关文档的覆盖比例',
    color: 'from-sakura-pink to-rose-300',
  },
  {
    label: 'MRR',
    value: '0.87',
    icon: '📈',
    description: '平均倒数排名 — 评估第一个相关结果出现在排序中的位置',
    color: 'from-sakura-accent to-sakura-light',
  },
  {
    label: 'Faithfulness',
    value: '0.94',
    icon: '✅',
    description: '忠实度 — 衡量生成回答与检索内容的语义一致性',
    color: 'from-sakura-pink to-orange-200',
  },
]
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="bg-white rounded-2xl shadow-sakura p-6">
      <h1 class="text-2xl font-bold text-sakura-text mb-1">📊 评估</h1>
      <p class="text-sakura-muted text-sm">检索质量与模型性能指标评估</p>
    </div>

    <!-- Metric cards grid -->
    <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      <div
        v-for="m in metrics"
        :key="m.label"
        class="bg-white rounded-2xl shadow-sakura overflow-hidden hover:shadow-lg hover:-translate-y-0.5 transition-all duration-300"
      >
        <!-- Gradient top bar -->
        <div :class="['h-2', 'bg-gradient-to-r', m.color]"></div>

        <!-- Card body -->
        <div class="p-5">
          <div class="flex items-center justify-between mb-3">
            <span class="text-2xl">{{ m.icon }}</span>
            <span class="text-[10px] bg-sakura-light/40 text-sakura-pink font-semibold px-2 py-0.5 rounded-full">
              即将上线
            </span>
          </div>

          <div class="text-4xl font-extrabold text-sakura-text mb-1 tracking-tight">
            {{ m.value }}
          </div>
          <p class="text-sm font-semibold text-sakura-pink mb-2">{{ m.label }}</p>
          <p class="text-xs text-sakura-muted leading-relaxed">{{ m.description }}</p>
        </div>
      </div>
    </div>

    <!-- More metrics coming note -->
    <div class="bg-white rounded-2xl shadow-sakura p-6 text-center">
      <div class="flex items-center justify-center gap-2">
        <span class="text-xl">🌸</span>
        <p class="text-sakura-muted font-medium">更多指标即将上线</p>
        <span class="text-xl">🌸</span>
      </div>
      <p class="text-xs text-sakura-muted/70 mt-1">
        敬请期待：NDCG、Precision@K、Recall@K、BLEU、ROUGE 等评估指标
      </p>
    </div>
  </div>
</template>
