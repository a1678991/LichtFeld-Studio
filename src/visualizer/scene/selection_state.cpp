/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scene/selection_state.hpp"
#include "visualizer/app_store.hpp"
#include <algorithm>
#include <cassert>

namespace lfs::vis {

    void SelectionState::selectNode(const core::NodeId id) {
        std::unique_lock lock(mutex_);
        selection_order_.assign(1, id);
        selected_lookup_.clear();
        selected_lookup_.insert(id);
        active_node_ = id;
        node_mask_dirty_ = true;
        bumpGeneration();
    }

    void SelectionState::selectNodes(const std::span<const core::NodeId> ids) {
        std::unique_lock lock(mutex_);
        selection_order_.clear();
        selected_lookup_.clear();
        for (const auto id : ids) {
            if (selected_lookup_.insert(id).second)
                selection_order_.push_back(id);
        }
        active_node_ = selection_order_.empty() ? core::NULL_NODE : selection_order_.back();
        node_mask_dirty_ = true;
        bumpGeneration();
    }

    void SelectionState::addToSelection(const core::NodeId id) {
        std::unique_lock lock(mutex_);
        if (selected_lookup_.insert(id).second)
            selection_order_.push_back(id);
        active_node_ = id;
        node_mask_dirty_ = true;
        bumpGeneration();
    }

    void SelectionState::removeFromSelection(const core::NodeId id) {
        std::unique_lock lock(mutex_);
        if (selected_lookup_.erase(id) > 0)
            std::erase(selection_order_, id);
        if (active_node_ == id)
            active_node_ = selection_order_.empty() ? core::NULL_NODE : selection_order_.back();
        node_mask_dirty_ = true;
        bumpGeneration();
    }

    void SelectionState::clearNodeSelection() {
        std::unique_lock lock(mutex_);
        selection_order_.clear();
        selected_lookup_.clear();
        active_node_ = core::NULL_NODE;
        node_mask_dirty_ = true;
        bumpGeneration();
    }

    bool SelectionState::isNodeSelected(const core::NodeId id) const {
        std::shared_lock lock(mutex_);
        return selected_lookup_.contains(id);
    }

    const std::vector<core::NodeId>& SelectionState::selectedNodeIds() const {
        return selection_order_;
    }

    core::NodeId SelectionState::activeNode() const {
        assert(active_node_ == core::NULL_NODE || selected_lookup_.contains(active_node_));
        return active_node_;
    }

    core::NodeId SelectionState::activeNodeId() const {
        std::shared_lock lock(mutex_);
        return active_node_;
    }

    size_t SelectionState::selectedNodeCount() const {
        std::shared_lock lock(mutex_);
        return selection_order_.size();
    }

    const std::vector<bool>& SelectionState::getNodeMask(const core::Scene& scene) const {
        std::shared_lock lock(mutex_);
        if (!node_mask_dirty_)
            return cached_node_mask_;

        // DCLP: release shared lock then acquire exclusive. Another thread may
        // rebuild the cache in the gap — the double-check below handles that.
        lock.unlock();
        std::unique_lock wlock(mutex_);

        if (!node_mask_dirty_)
            return cached_node_mask_;

        std::vector<std::string> names;
        names.reserve(selection_order_.size());
        for (const auto id : selection_order_) {
            const auto* node = scene.getNodeById(id);
            if (node)
                names.push_back(node->name);
        }

        cached_node_mask_ = scene.getSelectedNodeMask(names);
        node_mask_dirty_ = false;
        return cached_node_mask_;
    }

    void SelectionState::invalidateNodeMask() {
        std::unique_lock lock(mutex_);
        node_mask_dirty_ = true;
    }

    void SelectionState::bumpGeneration() {
        const uint32_t generation = generation_.fetch_add(1, std::memory_order_acq_rel) + 1;
        app_store().selection_generation.set(generation);
    }

} // namespace lfs::vis
