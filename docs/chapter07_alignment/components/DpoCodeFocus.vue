<script setup>
import { computed, ref } from 'vue'
import dpoCode from '../snippets/dpo-code-map.py?raw'

const props = defineProps({
  focus: {
    type: String,
    default: 'overview'
  },
  title: {
    type: String,
    default: ''
  }
})

const lines = dpoCode.trimEnd().split('\n')
const hovered = ref(false)
const pinned = ref(false)

const segments = [
  { id: 'A', label: '偏好数据', range: [5, 10] },
  { id: 'B', label: '序列 logprob', range: [13, 27] },
  { id: 'C', label: 'Policy / Reference', range: [30, 57] },
  { id: 'D', label: 'log-ratio', range: [59, 61] },
  { id: 'E', label: '隐式奖励差', range: [63, 67] },
  { id: 'F', label: 'DPO loss', range: [69, 78] },
  { id: 'G', label: '反向传播', range: [81, 89] },
  { id: 'H', label: '训练循环', range: [92, 100] }
]

const focusMap = {
  overview: {
    title: '完整 DPO 代码地图',
    active: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    compactRanges: [
      [5, 27],
      [30, 78],
      [81, 100]
    ],
    highlight: [
      7, 8, 9, 18, 20, 26, 27, 32, 38, 45, 46, 52, 59, 60, 61, 64, 65, 66, 67,
      70, 83, 85, 86, 87, 95, 96
    ]
  },
  data: {
    title: '偏好数据：prompt / chosen / rejected',
    active: ['A'],
    compactRanges: [[5, 10]],
    highlight: [7, 8, 9]
  },
  logprob: {
    title: '回答序列的 log probability',
    active: ['B'],
    compactRanges: [[13, 27]],
    highlight: [18, 20, 21, 26, 27]
  },
  models: {
    title: 'Policy 更新，Reference 冻结',
    active: ['C'],
    compactRanges: [[30, 57]],
    highlight: [32, 38, 45, 46, 52]
  },
  ratio: {
    title: 'DPO 的 log-ratio 与隐式奖励',
    active: ['D', 'E'],
    compactRanges: [[59, 67]],
    highlight: [60, 61, 64, 65, 66, 67]
  },
  loss: {
    title: 'DPO loss：-logsigmoid',
    active: ['E', 'F'],
    compactRanges: [[63, 78]],
    highlight: [64, 70, 75, 76]
  },
  train: {
    title: 'DPO 反向传播与训练循环',
    active: ['G', 'H'],
    compactRanges: [[81, 100]],
    highlight: [83, 85, 86, 87, 95, 96]
  }
}

const config = computed(() => focusMap[props.focus] || focusMap.overview)
const isExpanded = computed(() => hovered.value || pinned.value)
const activeTitle = computed(() => props.title || config.value.title)
const toggleLabel = computed(() => {
  if (pinned.value) return '收起完整代码'
  return isExpanded.value ? '固定完整代码' : '展开完整代码'
})

const highlighted = computed(() => new Set(config.value.highlight))
const activeSegments = computed(() => new Set(config.value.active))

function normalizeRanges(ranges) {
  return ranges
    .map(([start, end]) => [Math.max(1, start), Math.min(lines.length, end)])
    .filter(([start, end]) => start <= end)
    .sort((a, b) => a[0] - b[0])
}

const visibleRows = computed(() => {
  const ranges = isExpanded.value
    ? [[1, lines.length]]
    : normalizeRanges(config.value.compactRanges)
  const rows = []
  let previousEnd = 0

  for (const [start, end] of ranges) {
    if (previousEnd && start > previousEnd + 1) {
      rows.push({ type: 'gap', id: `${previousEnd}-${start}` })
    }

    for (let number = start; number <= end; number += 1) {
      rows.push({
        type: 'code',
        number,
        text: lines[number - 1],
        isHighlight: highlighted.value.has(number),
        isMarker: lines[number - 1].trimStart().startsWith('# [')
      })
    }

    previousEnd = end
  }

  return rows
})

function togglePinned() {
  pinned.value = !pinned.value
}
</script>

<template>
  <section
    class="dpo-code-focus"
    :class="{ 'is-expanded': isExpanded }"
    @mouseenter="hovered = true"
    @mouseleave="hovered = false"
  >
    <div class="dpo-code-focus-header" @click="togglePinned">
      <div class="dpo-code-focus-title">
        <span class="dpo-code-focus-kicker">DPO code lens</span>
        <strong>{{ activeTitle }}</strong>
      </div>
      <button
        class="dpo-code-focus-toggle"
        type="button"
        :aria-expanded="isExpanded"
        @click.stop="togglePinned"
      >
        {{ toggleLabel }}
      </button>
    </div>

    <div class="dpo-code-focus-segments" aria-label="DPO 代码结构">
      <span
        v-for="segment in segments"
        :key="segment.id"
        class="dpo-code-focus-segment"
        :class="{ 'is-active': activeSegments.has(segment.id) }"
      >
        <b>[{{ segment.id }}]</b>
        {{ segment.label }}
      </span>
    </div>

    <div class="dpo-code-focus-status">
      <span>{{ isExpanded ? '完整代码视图' : '局部重点视图' }}</span>
      <span>移入或点击可查看全局位置</span>
    </div>

    <pre class="dpo-code-focus-pre" tabindex="0"><code><template
      v-for="row in visibleRows"
      :key="row.type === 'gap' ? row.id : row.number"
    ><span v-if="row.type === 'gap'" class="dpo-code-focus-gap">        ⋮
</span><span
      v-else
      class="dpo-code-focus-line"
      :class="{
        'is-highlight': row.isHighlight,
        'is-marker': row.isMarker
      }"
    ><span class="dpo-code-focus-number">{{ String(row.number).padStart(3, ' ') }}</span><span class="dpo-code-focus-text">{{ row.text || ' ' }}</span>
</span></template></code></pre>
  </section>
</template>

<style scoped>
.dpo-code-focus {
  margin: 18px 0 26px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
  background: var(--vp-code-block-bg);
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
}

.dpo-code-focus-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 14px;
  border-bottom: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  cursor: pointer;
}

.dpo-code-focus-title {
  min-width: 0;
  display: grid;
  gap: 2px;
}

.dpo-code-focus-title strong {
  font-size: 14px;
  line-height: 1.35;
  color: var(--vp-c-text-1);
}

.dpo-code-focus-kicker {
  font-family: var(--vp-font-family-mono);
  font-size: 11px;
  line-height: 1.2;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
}

.dpo-code-focus-toggle {
  flex: 0 0 auto;
  min-height: 30px;
  padding: 0 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  color: var(--vp-c-brand-1);
  background: var(--vp-c-bg);
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
}

.dpo-code-focus-toggle:hover,
.dpo-code-focus-toggle:focus-visible {
  border-color: var(--vp-c-brand-1);
  outline: none;
}

.dpo-code-focus-segments {
  display: flex;
  gap: 6px;
  overflow-x: auto;
  padding: 10px 14px;
  border-bottom: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
}

.dpo-code-focus-segment {
  flex: 0 0 auto;
  padding: 4px 8px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-soft);
  font-size: 12px;
  line-height: 1.3;
}

.dpo-code-focus-segment.is-active {
  border-color: rgba(63, 81, 181, 0.38);
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
  font-weight: 700;
}

.dpo-code-focus-status {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 7px 14px;
  border-bottom: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-3);
  background: var(--vp-code-block-bg);
  font-size: 12px;
}

.dpo-code-focus-pre {
  max-height: 460px;
  margin: 0;
  padding: 10px 0;
  overflow: auto;
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  line-height: 1.55;
  background: var(--vp-code-block-bg);
}

.dpo-code-focus.is-expanded .dpo-code-focus-pre {
  max-height: 720px;
}

.dpo-code-focus-pre code {
  display: block;
  min-width: max-content;
  color: var(--vp-code-block-color);
}

.dpo-code-focus-line,
.dpo-code-focus-gap {
  display: grid;
  grid-template-columns: 48px minmax(0, 1fr);
  min-height: 20px;
  padding: 0 14px 0 0;
}

.dpo-code-focus-gap {
  display: block;
  padding-left: 54px;
  color: var(--vp-c-text-3);
  user-select: none;
}

.dpo-code-focus-number {
  padding-right: 12px;
  color: var(--vp-c-text-3);
  text-align: right;
  user-select: none;
}

.dpo-code-focus-text {
  white-space: pre;
}

.dpo-code-focus-line.is-marker {
  color: var(--vp-c-brand-1);
  font-weight: 700;
}

.dpo-code-focus-line.is-highlight {
  box-shadow: inset 3px 0 0 var(--vp-c-brand-1);
  background: rgba(63, 81, 181, 0.12);
  font-weight: 700;
}

.dark .dpo-code-focus {
  box-shadow: none;
}

.dark .dpo-code-focus-line.is-highlight {
  background: rgba(129, 140, 248, 0.16);
}

@media (max-width: 640px) {
  .dpo-code-focus-header,
  .dpo-code-focus-status {
    align-items: flex-start;
    flex-direction: column;
  }

  .dpo-code-focus-toggle {
    width: 100%;
  }

  .dpo-code-focus-pre {
    font-size: 12px;
  }
}
</style>
