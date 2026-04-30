<script setup>
import { computed, ref } from 'vue'
import ppoCode from '../snippets/ppo-code-map.py?raw'

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

const lines = ppoCode.trimEnd().split('\n')
const hovered = ref(false)
const pinned = ref(false)

const segments = [
  { id: 'A', label: '策略和值函数', range: [21, 27] },
  { id: 'B', label: '采样与 log_prob', range: [29, 42] },
  { id: 'C', label: '采样旧数据', range: [45, 74] },
  { id: 'D', label: '优势估计', range: [77, 92] },
  { id: 'E', label: 'PPO 更新', range: [95, 123] },
  { id: 'F', label: '训练循环', range: [126, 135] }
]

const focusMap = {
  overview: {
    title: '完整 PPO 代码地图',
    active: ['A', 'B', 'C', 'D', 'E', 'F'],
    compactRanges: [
      [21, 42],
      [77, 92],
      [109, 123],
      [126, 135]
    ],
    highlight: [
      24, 25, 31, 32, 33, 34, 86, 87, 90, 91, 110, 112, 113, 114, 115, 117, 119,
      121, 122, 123, 132, 133, 134
    ]
  },
  dist: {
    title: '动作分布 dist / log_prob',
    active: ['A', 'B'],
    compactRanges: [[21, 42]],
    highlight: [24, 25, 31, 32, 33, 34, 38, 39, 40, 41]
  },
  advantages: {
    title: '优势估计 advantages 与 value_loss',
    active: ['D', 'E'],
    compactRanges: [
      [77, 92],
      [117, 117]
    ],
    highlight: [86, 87, 90, 91, 117]
  },
  oldLogprobs: {
    title: '旧策略概率 old_logprobs',
    active: ['C'],
    compactRanges: [[45, 74]],
    highlight: [52, 53, 62, 72]
  },
  ratio: {
    title: '策略比率 ratio',
    active: ['C', 'E'],
    compactRanges: [[95, 112]],
    highlight: [101, 109, 110, 112]
  },
  surr1: {
    title: '未裁剪代理目标 surr1',
    active: ['E'],
    compactRanges: [[109, 115]],
    highlight: [112]
  },
  clip: {
    title: 'PPO-Clip 更新核心',
    active: ['E'],
    compactRanges: [[109, 119]],
    highlight: [110, 112, 113, 114, 115]
  },
  loss: {
    title: '总 loss 与反向传播',
    active: ['E'],
    compactRanges: [[117, 123]],
    highlight: [117, 118, 119, 121, 122, 123]
  },
  train: {
    title: 'PPO 训练循环',
    active: ['F'],
    compactRanges: [[126, 135]],
    highlight: [132, 133, 134]
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
    class="ppo-code-focus"
    :class="{ 'is-expanded': isExpanded }"
    @mouseenter="hovered = true"
    @mouseleave="hovered = false"
  >
    <div class="ppo-code-focus-header" @click="togglePinned">
      <div class="ppo-code-focus-title">
        <span class="ppo-code-focus-kicker">PPO code lens</span>
        <strong>{{ activeTitle }}</strong>
      </div>
      <button
        class="ppo-code-focus-toggle"
        type="button"
        :aria-expanded="isExpanded"
        @click.stop="togglePinned"
      >
        {{ toggleLabel }}
      </button>
    </div>

    <div class="ppo-code-focus-segments" aria-label="PPO 代码结构">
      <span
        v-for="segment in segments"
        :key="segment.id"
        class="ppo-code-focus-segment"
        :class="{ 'is-active': activeSegments.has(segment.id) }"
      >
        <b>[{{ segment.id }}]</b>
        {{ segment.label }}
      </span>
    </div>

    <div class="ppo-code-focus-status">
      <span>{{ isExpanded ? '完整代码视图' : '局部重点视图' }}</span>
      <span>移入或点击可查看全局位置</span>
    </div>

    <pre class="ppo-code-focus-pre" tabindex="0"><code><template
      v-for="row in visibleRows"
      :key="row.type === 'gap' ? row.id : row.number"
    ><span v-if="row.type === 'gap'" class="ppo-code-focus-gap">        ⋮
</span><span
      v-else
      class="ppo-code-focus-line"
      :class="{
        'is-highlight': row.isHighlight,
        'is-marker': row.isMarker
      }"
    ><span class="ppo-code-focus-number">{{ String(row.number).padStart(3, ' ') }}</span><span class="ppo-code-focus-text">{{ row.text || ' ' }}</span>
</span></template></code></pre>
  </section>
</template>

<style scoped>
.ppo-code-focus {
  margin: 18px 0 26px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
  background: var(--vp-code-block-bg);
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
}

.ppo-code-focus-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 14px;
  border-bottom: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  cursor: pointer;
}

.ppo-code-focus-title {
  min-width: 0;
  display: grid;
  gap: 2px;
}

.ppo-code-focus-title strong {
  font-size: 14px;
  line-height: 1.35;
  color: var(--vp-c-text-1);
}

.ppo-code-focus-kicker {
  font-family: var(--vp-font-family-mono);
  font-size: 11px;
  line-height: 1.2;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
}

.ppo-code-focus-toggle {
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

.ppo-code-focus-toggle:hover,
.ppo-code-focus-toggle:focus-visible {
  border-color: var(--vp-c-brand-1);
  outline: none;
}

.ppo-code-focus-segments {
  display: flex;
  gap: 6px;
  overflow-x: auto;
  padding: 10px 14px;
  border-bottom: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
}

.ppo-code-focus-segment {
  flex: 0 0 auto;
  padding: 4px 8px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-soft);
  font-size: 12px;
  line-height: 1.3;
}

.ppo-code-focus-segment.is-active {
  border-color: rgba(63, 81, 181, 0.38);
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
  font-weight: 700;
}

.ppo-code-focus-status {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 7px 14px;
  border-bottom: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-3);
  background: var(--vp-code-block-bg);
  font-size: 12px;
}

.ppo-code-focus-pre {
  max-height: 460px;
  margin: 0;
  padding: 10px 0;
  overflow: auto;
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  line-height: 1.55;
  background: var(--vp-code-block-bg);
}

.ppo-code-focus.is-expanded .ppo-code-focus-pre {
  max-height: 720px;
}

.ppo-code-focus-pre code {
  display: block;
  min-width: max-content;
  color: var(--vp-code-block-color);
}

.ppo-code-focus-line,
.ppo-code-focus-gap {
  display: grid;
  grid-template-columns: 48px minmax(0, 1fr);
  min-height: 20px;
  padding: 0 14px 0 0;
}

.ppo-code-focus-gap {
  display: block;
  padding-left: 54px;
  color: var(--vp-c-text-3);
  user-select: none;
}

.ppo-code-focus-number {
  padding-right: 12px;
  color: var(--vp-c-text-3);
  text-align: right;
  user-select: none;
}

.ppo-code-focus-text {
  white-space: pre;
}

.ppo-code-focus-line.is-marker {
  color: var(--vp-c-brand-1);
  font-weight: 700;
}

.ppo-code-focus-line.is-highlight {
  box-shadow: inset 3px 0 0 var(--vp-c-brand-1);
  background: rgba(63, 81, 181, 0.12);
  font-weight: 700;
}

.dark .ppo-code-focus {
  box-shadow: none;
}

.dark .ppo-code-focus-line.is-highlight {
  background: rgba(129, 140, 248, 0.16);
}

@media (max-width: 640px) {
  .ppo-code-focus-header,
  .ppo-code-focus-status {
    align-items: flex-start;
    flex-direction: column;
  }

  .ppo-code-focus-toggle {
    width: 100%;
  }

  .ppo-code-focus-pre {
    font-size: 12px;
  }
}
</style>
