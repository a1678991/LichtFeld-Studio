/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_uilist.hpp"
#include "core/logger.hpp"

namespace lfs::python {

    PyUIListRegistry& PyUIListRegistry::instance() {
        static PyUIListRegistry registry;
        return registry;
    }

    void PyUIListRegistry::register_uilist(nb::object list_class) {
        std::lock_guard lock(mutex_);

        std::string id;
        if (nb::hasattr(list_class, "list_id")) {
            id = nb::cast<std::string>(list_class.attr("list_id"));
        } else if (nb::hasattr(list_class, "__name__")) {
            id = nb::cast<std::string>(list_class.attr("__name__"));
        } else {
            LOG_ERROR("UIList class missing list_id or __name__ attribute");
            return;
        }

        uilists_[id] = {id, list_class, nb::object()};
    }

    void PyUIListRegistry::unregister_uilist(const std::string& id) {
        std::lock_guard lock(mutex_);
        uilists_.erase(id);
    }

    void PyUIListRegistry::unregister_all() {
        std::lock_guard lock(mutex_);
        uilists_.clear();
    }

    PyUIListInfo* PyUIListRegistry::ensure_instance(PyUIListInfo& uilist) {
        if (!uilist.list_instance.is_valid() || uilist.list_instance.is_none()) {
            nb::gil_scoped_acquire gil;
            try {
                uilist.list_instance = uilist.list_class();
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to instantiate UIList {}: {}", uilist.id, e.what());
                return nullptr;
            }
        }
        return &uilist;
    }

    PyUIListInfo* PyUIListRegistry::get_uilist(const std::string& id) {
        std::lock_guard lock(mutex_);
        auto it = uilists_.find(id);
        if (it == uilists_.end())
            return nullptr;
        return ensure_instance(it->second);
    }

    std::vector<std::string> PyUIListRegistry::get_uilist_ids() const {
        std::lock_guard lock(mutex_);
        std::vector<std::string> ids;
        ids.reserve(uilists_.size());
        for (const auto& [id, _] : uilists_) {
            ids.push_back(id);
        }
        return ids;
    }

    std::tuple<bool, int> PyUILayoutTemplates::template_list(
        PyUILayout& layout,
        const std::string& listtype_name,
        const std::string& list_id,
        nb::object data,
        const std::string& propname,
        int active_index,
        int rows) {

        bool changed = false;
        int new_index = active_index;

        nb::gil_scoped_acquire gil;

        nb::object list_obj;
        try {
            if (nb::hasattr(data, propname.c_str())) {
                list_obj = data.attr(propname.c_str());
            } else if (nb::hasattr(data, "get")) {
                list_obj = data.attr("get")(propname);
            } else {
                LOG_ERROR("template_list: data object has no property '{}'", propname);
                return {false, active_index};
            }
        } catch (const std::exception& e) {
            LOG_ERROR("template_list: failed to get property '{}': {}", propname, e.what());
            return {false, active_index};
        }

        size_t list_len = 0;
        try {
            list_len = nb::len(list_obj);
        } catch (const std::exception& e) {
            LOG_ERROR("template_list: failed to get list length: {}", e.what());
            return {false, active_index};
        }

        (void)layout;
        (void)listtype_name;
        (void)list_id;
        (void)rows;
        (void)list_len;
        return {changed, new_index};
    }

    bool PyUILayoutTemplates::template_tree(
        PyUILayout& layout,
        const std::string& label,
        nb::object draw_callback,
        bool default_open) {

        (void)label;
        const bool is_open = default_open;
        if (is_open) {
            nb::gil_scoped_acquire gil;
            try {
                draw_callback(layout);
            } catch (const std::exception& e) {
                LOG_ERROR("template_tree draw callback: {}", e.what());
            }
        }
        return is_open;
    }

    std::tuple<bool, std::string> PyUILayoutTemplates::template_id(
        PyUILayout& layout,
        const std::string& label,
        const std::vector<std::string>& items,
        const std::string& current_id) {

        int current_idx = -1;
        for (size_t i = 0; i < items.size(); ++i) {
            if (items[i] == current_id) {
                current_idx = static_cast<int>(i);
                break;
            }
        }

        (void)layout;
        (void)label;
        (void)current_idx;
        return {false, current_id};
    }

    void register_uilist(nb::module_& m) {
        m.def(
            "register_uilist", [](nb::object list_class) {
                PyUIListRegistry::instance().register_uilist(list_class);
            },
            nb::arg("list_class"), "Register a UIList class for custom list rendering");

        m.def(
            "unregister_uilist", [](const std::string& id) {
                PyUIListRegistry::instance().unregister_uilist(id);
            },
            nb::arg("id"), "Unregister a UIList by ID");

        m.def(
            "unregister_all_uilists", []() { PyUIListRegistry::instance().unregister_all(); },
            "Unregister all UIList classes");
        m.def(
            "get_uilist_ids", []() { return PyUIListRegistry::instance().get_uilist_ids(); },
            "Get all registered UIList IDs");
    }

    void add_template_methods_to_uilayout(nb::class_<PyUILayout>& layout_class) {
        layout_class
            .def(
                "template_list",
                [](PyUILayout& self, const std::string& listtype_name, const std::string& list_id,
                   nb::object data, const std::string& propname, int active_index, int rows) {
                    return PyUILayoutTemplates::template_list(self, listtype_name, list_id,
                                                              data, propname, active_index, rows);
                },
                nb::arg("listtype_name"), nb::arg("list_id"), nb::arg("data"),
                nb::arg("propname"), nb::arg("active_index"), nb::arg("rows") = 5,
                "Draw a selectable list widget, returns (changed, new_active_index)")
            .def(
                "template_tree",
                [](PyUILayout& self, const std::string& label, nb::object draw_callback, bool default_open) {
                    return PyUILayoutTemplates::template_tree(self, label, draw_callback, default_open);
                },
                nb::arg("label"), nb::arg("draw_callback"), nb::arg("default_open") = false,
                "Draw a collapsible tree node with custom content")
            .def(
                "template_id",
                [](PyUILayout& self, const std::string& label, const std::vector<std::string>& items,
                   const std::string& current_id) {
                    return PyUILayoutTemplates::template_id(self, label, items, current_id);
                },
                nb::arg("label"), nb::arg("items"), nb::arg("current_id"),
                "Draw a combo selector from string IDs, returns (changed, selected_id)");
    }

} // namespace lfs::python
