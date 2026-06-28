<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { listDocuments, uploadDocument, deleteDocument } from '../api'
import DocUploader from '../components/DocUploader.vue'

interface Doc {
  doc_id: string
  filename: string
  chunk_count: number
  uploaded_at: string
  collection?: string
}

const docs = ref<Doc[]>([])
const loading = ref(false)
const uploading = ref(false)
const error = ref('')

async function loadDocs() {
  loading.value = true
  error.value = ''
  try {
    const data = await listDocuments()
    docs.value = Array.isArray(data) ? data : (data.documents || [])
  } catch (e: any) {
    error.value = e.message || '加载文档列表失败'
    docs.value = []
  } finally {
    loading.value = false
  }
}

async function handleUpload(file: File) {
  uploading.value = true
  error.value = ''
  try {
    await uploadDocument(file)
    await loadDocs()
  } catch (e: any) {
    error.value = e.message || '上传失败，请检查后端是否已启动'
  } finally {
    uploading.value = false
  }
}

async function handleDelete(docId: string, filename: string) {
  if (!confirm(`确定要删除 "${filename}" 吗？此操作不可撤销。`)) return
  try {
    await deleteDocument(docId)
    docs.value = docs.value.filter(d => d.doc_id !== docId)
  } catch (e: any) {
    error.value = e.message || '删除失败'
  }
}

function formatDate(dateStr: string): string {
  if (!dateStr) return ''
  try {
    const d = new Date(dateStr)
    return d.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' }) +
      ' ' + d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  } catch {
    return dateStr
  }
}

onMounted(loadDocs)
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="bg-white rounded-2xl shadow-sakura p-6">
      <h1 class="text-2xl font-bold text-sakura-text mb-1">📁 文档管理</h1>
      <p class="text-sakura-muted text-sm">上传、浏览与管理知识库文档</p>
    </div>

    <!-- Error banner -->
    <div
      v-if="error"
      class="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700 flex items-center justify-between"
    >
      <span>{{ error }}</span>
      <button @click="error = ''" class="text-red-400 hover:text-red-600 font-bold text-lg leading-none">&times;</button>
    </div>

    <!-- Upload area -->
    <DocUploader @upload="handleUpload" />

    <!-- Upload loading -->
    <div
      v-if="uploading"
      class="bg-white rounded-2xl shadow-sakura p-6 flex items-center gap-3"
    >
      <div class="w-5 h-5 border-2 border-sakura-light border-t-sakura-pink rounded-full animate-spin"></div>
      <span class="text-sakura-muted">正在上传并处理文档...</span>
    </div>

    <!-- Document list -->
    <div class="bg-white rounded-2xl shadow-sakura p-6">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-lg font-semibold text-sakura-text">
          文档列表
          <span class="text-sm text-sakura-muted font-normal ml-2">共 {{ docs.length }} 个</span>
        </h2>
        <button
          @click="loadDocs"
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
        v-else-if="docs.length === 0"
        class="flex flex-col items-center py-10 text-center"
      >
        <div class="text-5xl mb-3">📭</div>
        <p class="text-sakura-muted font-medium">还没有上传任何文档</p>
        <p class="text-sakura-muted/70 text-sm mt-1">使用上方的上传区域添加文档吧~</p>
      </div>

      <!-- Document cards -->
      <div v-else class="grid gap-3">
        <div
          v-for="doc in docs"
          :key="doc.doc_id"
          class="flex items-center justify-between bg-sakura-bg rounded-xl px-4 py-3 hover:shadow-sakura transition-shadow group"
        >
          <!-- Left: icon + info -->
          <div class="flex items-center gap-3 min-w-0">
            <div class="text-2xl shrink-0">📄</div>
            <div class="min-w-0">
              <p class="text-sakura-text font-medium truncate" :title="doc.filename">
                {{ doc.filename }}
              </p>
              <div class="flex items-center gap-3 text-xs text-sakura-muted mt-0.5">
                <span>{{ doc.chunk_count ?? '?' }} 个分块</span>
                <span v-if="doc.uploaded_at">{{ formatDate(doc.uploaded_at) }}</span>
                <span v-if="doc.collection" class="bg-sakura-light/30 px-1.5 py-0.5 rounded text-sakura-text/70">
                  {{ doc.collection }}
                </span>
              </div>
            </div>
          </div>

          <!-- Right: delete button -->
          <button
            @click="handleDelete(doc.doc_id, doc.filename)"
            class="text-sakura-muted/60 hover:text-red-400 hover:bg-red-50 p-2 rounded-lg transition-all opacity-0 group-hover:opacity-100 shrink-0"
            title="删除文档"
          >
            🗑️
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
