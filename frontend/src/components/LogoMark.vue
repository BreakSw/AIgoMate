<script setup lang="ts">
withDefaults(defineProps<{
  size?: 'small' | 'medium' | 'large'
}>(), {
  size: 'medium',
})
</script>

<template>
  <span class="logo-symbol" :class="`logo-symbol--${size}`" aria-hidden="true">
    <span class="logo-orbit"></span>
    <span class="logo-branch logo-branch--upper"></span>
    <span class="logo-branch logo-branch--lower"></span>
    <span class="logo-node logo-node--origin"></span>
    <span class="logo-node logo-node--upper"></span>
    <span class="logo-node logo-node--lower"></span>
  </span>
</template>

<style scoped>
.logo-symbol {
  position: relative;
  display: inline-block;
  flex: 0 0 auto;
  width: 24px;
  height: 24px;
  color: inherit;
}

.logo-symbol--small { width: 18px; height: 18px; }
.logo-symbol--large { width: 30px; height: 30px; }

.logo-orbit {
  position: absolute;
  left: 22%;
  top: 22%;
  width: 43%;
  height: 43%;
  border: 1.8px solid currentColor;
  border-radius: 50%;
  opacity: .72;
}

.logo-branch {
  position: absolute;
  left: 48%;
  width: 34%;
  height: 2px;
  border-radius: 99px;
  background: currentColor;
  transform-origin: left center;
}

.logo-branch--upper { top: 43%; transform: rotate(-37deg); }
.logo-branch--lower { top: 58%; transform: rotate(31deg); }

.logo-node {
  position: absolute;
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 0 2px color-mix(in srgb, currentColor 18%, transparent);
}

.logo-node--origin { left: 16%; top: 49%; width: 6px; height: 6px; }
.logo-node--upper { right: 10%; top: 16%; }
.logo-node--lower { right: 8%; bottom: 13%; }
</style>
