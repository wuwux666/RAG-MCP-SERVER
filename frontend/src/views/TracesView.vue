<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { listQueryTraces, listIngestionTraces } from '../api'
import TraceTimeline from '../components/TraceTimeline.vue'

interface TraceStage {
  name: string
  elapsed_ms: number
}

interface Trace {
  trace_id: string
  query?: string
  metadata?: string
  filename?: string
  stages?: TraceStage[]
  elapsed_ms?: number
  created_at?: string
}

const activeTab = ref<'query' | 'ingestion'>('query')
const queryTraces = ref<Trace[]>([])
const ingestionTraces = ref<Trace[]>([])
const loading = ref(false)
const error = ref('')

const traces = computed(() => {
  return activeTab.value === 'query' ? queryTraces.value : ingestionTraces.value
})

async function loadQueryTraces() {
  try {
    const data = await listQueryTraces(50)
    queryTraces.value = Array.isArray(data) ? data : (data.traces || [])
  } catch (e: any) {
    queryTraces.value = []
    throw e
  }
}

async function loadIngestionTraces() {
  try {
    const data = await listIngestionTraces(20)
    ingestionTraces.value = Array.isArray(data) ? data : (data.traces || [])
  } catch (e: any) {
    ingestionTraces.value = []
    throw e
  }
}

async function loadTraces() {
  loading.value = true
  error.value = ''
  try {
    if (activeTab.value === 'query') {
      await loadQueryTraces()
    } else {
      await loadIngestionTraces()
    }
  } catch (e: any) {
    error.value = e.message || '加载追踪数据失败'
  } finally {
    loading.value = false
  }
}

function switchTab(tab: 'query' | 'ingestion') {
  activeTab.value = tab
  if (tab === 'query' && queryTraces.value.length === 0) loadTraces()
  if (tab === 'ingestion' && ingestionTraces.value.length === 0) loadTraces()
}

function formatDate(dateStr: string): string {
  if (!dateStr) return ''
  try {
    const d = new Date(dateStr)
    return d.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' }) +
      ' ' + d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  } catch {
    return dateStr
  }
}

function totalElapsed(stages?: TraceStage[]): number {
  if (!stages?.length) return 0
  return stages.reduce((sum, s) => sum + s.elapsed_ms, 0)
}

onMounted(loadTraces)
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="bg-white rounded-2xl shadow-sakura p-6">
      <h1 class="text-2xl font-bold text-sakura-text mb-1">🔍 追踪记录</h1>
      <p class="text-sakura-muted text-sm">监控查询与摄取流程的链路追踪数据</p>
    </div>

    <!-- Error banner -->
    <div
      v-if="error"
      class="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700 flex items-center justify-between"
    >
      <span>{{ error }}</span>
      <button @click="error = ''" class="text-red-400 hover:text-red-600 font-bold text-lg leading-none">&times;</button>
    </div>

    <!-- Tabs -->
    <div class="flex gap-2">
      <button
        @click="switchTab('query')"
        class="px-5 py-2.5 rounded-xl font-medium text-sm transition-all duration-200"
        :class="activeTab === 'query'
          ? 'bg-sakura-pink text-white shadow-sakura'
          : 'bg-white text-sakura-muted hover:bg-sakura-light/20'"
      >
        💬 查询追踪
      </button>
      <button
        @click="switchTab('ingestion')"
        class="px-5 py-2.5 rounded-xl font-medium text-sm transition-all duration-200"
        :class="activeTab === 'ingestion'
          ? 'bg-sakura-pink text-white shadow-sakura'
          : 'bg-white text-sakura-muted hover:bg-sakura-light/20'"
      >
        📥 摄取追踪
      </button>
    </div>

    <!-- Trace list -->
    <div class="bg-white rounded-2xl shadow-sakura p-6">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-lg font-semibold text-sakura-text">
          {{ activeTab === 'query' ? '查询追踪' : '摄取追踪' }}
          <span class="text-sm text-sakura-muted font-normal ml-2">共 {{ traces.length }} 条</span>
        </h2>
        <button
          @click="loadTraces"
          :disabled="loading"
          class="text-sm text-sakura-pink hover:text-sakura-accent transition-colors"
        >
          🔄 刷新
        </button>
      </div>

      <!-- Loading state -->
      <div v-if="loading" class="flex justify-center py-10">
        <div class="w-8 h-8 border-4 border-sakura-light border-t-sakura-pink rounded-full animate-spin"></div>
      </div>

      <!-- Empty state -->
      <div
        v-else-if="traces.length === 0"
        class="flex flex-col items-center py-10 text-center"
      >
        <div class="text-5xl mb-3">🔍</div>
        <p class="text-sakura-muted font-medium">暂无追踪记录</p>
        <p class="text-sakura-muted/70 text-sm mt-1">
          {{ activeTab === 'query' ? '进行一次查询后将在此处显示追踪数据' : '上传并处理文档后将在此处显示摄取追踪' }}
        </p>
      </div>

      <!-- Trace cards -->
      <div v-else class="space-y-4">
        <div
          v-for="trace in traces"
          :key="trace.trace_id"
          class="bg-sakura-bg rounded-xl p-4 hover:shadow-sakura transition-shadow"
        >
          <!-- Trace header -->
          <div class="flex items-start justify-between mb-3">
            <div class="flex-1 min-w-0">
              <!-- Query trace: show the query text -->
              <p
                v-if="activeTab === 'query' && trace.query"
                class="text-sakura-text font-medium truncate"
                :title="trace.query"
              >
                💬 {{ trace.query }}
              </p>
              <!-- Ingestion trace: show filename -->
              <p
                v-else-if="activeTab === 'ingestion'"
                class="text-sakura-text font-medium truncate"
                :title="trace.filename || trace.metadata"
              >
                📄 {{ trace.filename || trace.metadata || '未知文件' }}
              </p>
              <p class="text-xs text-sakura-muted mt-0.5">
                <code class="bg-sakura-light/20 px-1 rounded text-[11px]">{{ trace.trace_id }}</code>
                <span v-if="trace.created_at" class="ml-2">{{ formatDate(trace.created_at) }}</span>
              </p>
            </div>
            <!-- Total elapsed -->
            <div class="text-right shrink-0 ml-3">
              <div class="text-lg font-bold text-sakura-pink">
                {{ trace.elapsed_ms ?? totalElapsed(trace.stages) }}<span class="text-xs font-normal">ms</span>
              </div>
            </div>
          </div>

          <!-- Stages timeline -->
          <TraceTimeline
            v-if="trace.stages?.length"
            :stages="trace.stages"
          />
          <p v-else class="text-xs text-sakura-muted italic">无阶段详情</p>
        </div>
      </div>
    </div>
  </div>
</template>
