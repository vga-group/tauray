/*
The MIT License (MIT)

Copyright (c) 2020, 2021, 2022 Julius Ikkala

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
/** \mainpage MonkeroECS
 *
 * \section intro_sec Introduction
 *
 * MonkeroECS is a fairly small, header-only Entity Component System rescued
 * from a game engine. It aims for simple and terse usage and also contains an
 * event system.
 *
 * MonkeroECS is written in C++17 and the only dependency is the standard
 * library. Note that the code isn't pretty and has to do some pretty gnarly
 * trickery to make the usable as terse as it is.
 *
 * While performance is one of the goals of this ECS, it was more written with
 * flexibility in mind. Particularly, random access and multi-component
 * iteration should be quite fast. All components have stable storage, that is,
 * they will not get moved around after their creation. Therefore, the ECS also
 * works with components that are not copyable or movable and pointers to them
 * will not be invalidated until the component is removed.
 *
 * Adding the ECS to your project is simple; just copy the monkeroecs.hh yo
 * your codebase and include it!
 *
 * The name is a reference to the game the ECS was originally created for, but
 * it was unfortunately never finished or released in any form despite the
 * engine being in a finished state.
 */
#ifndef MONKERO_ECS_HH
#define MONKERO_ECS_HH
//#define MONKERO_CONTAINER_DEALLOCATE_BUCKETS
//#define MONKERO_CONTAINER_DEBUG_UTILS
#include <cstdint>
#include <map>
#include <functional>
#include <memory>
#include <vector>
#include <limits>
#include <utility>
#include <type_traits>
#include <algorithm>
#include <tuple>
#include <map>
#include <cstring>
#ifdef MONKERO_CONTAINER_DEBUG_UTILS
#include <iostream>
#include <bitset>
#endif

// Thanks, MSVC -.-
#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

/** This namespace contains all of MonkeroECS. */
namespace monkero
{

/** The entity type, it's just an ID.
 * An entity alone will not take up memory in the ECS, only once components are
 * added does the entity truly use memory. You can change this to uint64_t if
 * you truly need over 4 billion entities and have tons of memory.
 */
using entity = std::uint32_t;
// You are not allowed to use this entity ID.
inline constexpr entity INVALID_ENTITY = 0;

class scene;

/** A built-in event emitted when a component is added to the ECS. */
template<typename Component>
struct add_component
{
    entity id; /**< The entity that got the component */
    Component* data; /**< A pointer to the component */
};

/** A built-in event emitted when a component is removed from the ECS.
 * The destructor of class scene will emit remove_component events for all
 * components still left at that point.
 */
template<typename Component>
struct remove_component
{
    entity id; /**< The entity that lost the component */
    Component* data; /**< A pointer to the component (it's not destroyed quite yet) */
};

/** This class is used to receive events of the specified type(s).
 * Once it is destructed, no events will be delivered to the associated
 * callback function anymore.
 * \note Due to the callback nature, event subscriptions are immovable.
 * \see receiver
 */
class event_subscription
{
friend class scene;
public:
    inline event_subscription(scene* ctx = nullptr, std::size_t subscription_id = 0);
    inline explicit event_subscription(event_subscription&& other);
    event_subscription(const event_subscription& other) = delete;
    inline ~event_subscription();

    event_subscription& operator=(event_subscription&& other) = delete;
    event_subscription& operator=(const event_subscription& other) = delete;

private:
    scene* ctx;
    std::size_t subscription_id;
};

// Provides event receiving facilities for one type alone. Derive
// from class receiver instead in your own code.
template<typename EventType>
class event_receiver
{
public:
    virtual ~event_receiver() = default;

    /** Receivers must implement this for each received event type.
     * It's called by emitters when an event of EventType is emitted.
     * \param ctx The ECS this receiver is part of.
     * \param event The event that occurred.
     */
    virtual void handle(scene& ctx, const EventType& event) = 0;
};

/** Deriving from this class allows systems to receive events of the specified
 * type(s). Just give a list of the desired event types in the template
 * parameters and the system will start receiving them from all other systems
 * automatically.
 */
template<typename... ReceiveEvents>
class receiver: public event_receiver<ReceiveEvents>...
{
friend class scene;
private:
    event_subscription sub;
};

/** Specializing this class for your component type and implementing the given
 * functions allows for accelerated entity searching based on any parameter you
 * want to define. The default does not define any searching operations.
 */
template<typename Component>
class search_index
{
public:
    // It's up to you how you want to do this searching. You should return
    // INVALID_ENTITY if there was no entity matching the search parameters.
    // You can also have multiple find() overloads for the same component type.

    // entity find(some parameters here) const;

    /** Called automatically when an entity of this component type is added.
     * \param id ID of the entity whose component is being added.
     * \param data the component data itself.
     */
    void add_entity(entity id, const Component& data);

    /** Called automatically when an entity of this component type is removed.
     * \param id ID of the entity whose component is being removed.
     * \param data the component data itself.
     */
    void remove_entity(entity id, const Component& data);

    /** Manual full search index refresh.
     * Never called automatically, the ECS has a refresh_search_index() function
     * that the user must call that then calls this.
     * \param e the ECS context.
     */
    void update(scene& e);

    // Don't copy this one though, or else you won't get some add_entity or
    // remove_entity calls.
    using empty_default_impl = void;
};

template<typename T, typename=void>
struct has_bucket_exp_hint: std::false_type { };

template<typename T>
struct has_bucket_exp_hint<
    T,
    decltype((void)
        T::bucket_exp_hint, void()
    )
> : std::is_integral<decltype(T::bucket_exp_hint)> { };

/** Provides bucket size choice for the component container.
 * To change the number of entries in a bucket for your component type, you have
 * two options:
 * A (preferred when you can modify the component type):
 *     Add a `static constexpr std::uint32_t bucket_exp_hint = N;`
 * B (needed when you cannot modify the component type):
 *     Specialize component_bucket_exp_hint for your type and provide a
 *     `static constexpr std::uint32_t value = N;`
 * Where the bucket size will then be 2**N.
 */
template<typename T>
struct component_bucket_exp_hint
{
    static constexpr std::uint32_t value = []{
        if constexpr (has_bucket_exp_hint<T>::value)
        {
            return T::bucket_exp_hint;
        }
        else
        {
            uint32_t i = 6;
            // Aim for 65kb buckets
            while((std::max(sizeof(T), std::uint64_t(4))<<i) < std::uint64_t(65536))
                ++i;
            return i;
        }
    }();
};

class component_container_base
{
public:
    virtual ~component_container_base() = default;

    inline virtual void start_batch() = 0;
    inline virtual void finish_batch() = 0;
    inline virtual void erase(entity id) = 0;
    inline virtual void clear() = 0;
    inline virtual std::size_t size() const = 0;
    inline virtual void update_search_index() = 0;
    inline virtual void list_entities(
        std::map<entity, entity>& translation_table
    ) = 0;
    inline virtual void concat(
        scene& target,
        const std::map<entity, entity>& translation_table
    ) = 0;
    inline virtual void copy(
        scene& target,
        entity result_id,
        entity original_id
    ) = 0;
};

struct component_container_entity_advancer
{
public:
    inline void advance();

    std::uint32_t bucket_mask;
    std::uint32_t bucket_exp;
    entity*** bucket_jump_table;
    std::uint32_t current_bucket;
    entity current_entity;
    entity* current_jump_table;
};

template<typename T>
class component_container: public component_container_base
{
public:
    using bitmask_type = std::uint64_t;
    static constexpr uint32_t bitmask_bits = 64;
    static constexpr uint32_t bitmask_shift = 6; // 64 = 2**6
    static constexpr uint32_t bitmask_mask = 0x3F;
    static constexpr uint32_t initial_bucket_count = 16u;
    static constexpr bool tag_component = std::is_empty_v<T>;
    static constexpr std::uint32_t bucket_exp =
        component_bucket_exp_hint<T>::value;
    static constexpr std::uint32_t bucket_mask = (1u<<bucket_exp)-1;
    static constexpr std::uint32_t bucket_bitmask_units =
        std::max(1u, (1u<<bucket_exp)>>bitmask_shift);

    component_container(scene& ctx);
    component_container(component_container&& other) = delete;
    component_container(const component_container& other) = delete;
    ~component_container();

    T* operator[](entity e);
    const T* operator[](entity e) const;

    void insert(entity id, T&& value);

    template<typename... Args>
    void emplace(entity id, Args&&... value);

    void erase(entity id) override;

    void clear() override;

    bool contains(entity id) const;

    void start_batch() override;
    void finish_batch() override;

    class iterator
    {
    friend class component_container<T>;
    public:
        using component_type = T;

        iterator() = delete;
        iterator(const iterator& other) = default;

        iterator& operator++();
        iterator operator++(int);
        std::pair<entity, T*> operator*();
        std::pair<entity, const T*> operator*() const;

        bool operator==(const iterator& other) const;
        bool operator!=(const iterator& other) const;

        bool try_advance(entity id);

        operator bool() const;
        entity get_id() const;
        component_container<T>* get_container() const;
        component_container_entity_advancer get_advancer();

    private:
        iterator(component_container& from, entity e);

        component_container* from;
        entity current_entity;
        std::uint32_t current_bucket;
        entity* current_jump_table;
        T* current_components;
    };

    iterator begin();
    iterator end();
    std::size_t size() const override;

    void update_search_index() override;

    void list_entities(
        std::map<entity, entity>& translation_table
    ) override;
    void concat(
        scene& target,
        const std::map<entity, entity>& translation_table
    ) override;
    void copy(
        scene& target,
        entity result_id,
        entity original_id
    ) override;

    template<typename... Args>
    entity find_entity(Args&&... args) const;

#ifdef MONKERO_CONTAINER_DEBUG_UTILS
    bool test_invariant() const;
    void print_bitmask() const;
    void print_jump_table() const;
#endif

private:
    T* get_unsafe(entity e);
    void destroy();
    void jump_table_insert(entity id);
    void jump_table_erase(entity id);
    std::size_t get_top_bitmask_size() const;
    bool bitmask_empty(std::uint32_t bucket_index) const;
    void bitmask_insert(entity id);
    // Returns a hint to whether the whole bucket should be removed or not.
    bool bitmask_erase(entity id);

    template<typename... Args>
    void bucket_insert(entity id, Args&&... args);
    void bucket_erase(entity id, bool signal);
    void bucket_self_erase(std::uint32_t bucket_index);
    void try_jump_table_bucket_erase(std::uint32_t bucket_index);
    void ensure_bucket_space(entity id);
    void ensure_bitmask(std::uint32_t bucket_index);
    void ensure_jump_table(std::uint32_t bucket_index);
    bool batch_change(entity id);
    entity find_previous_entity(entity id);
    void signal_add(entity id, T* data);
    void signal_remove(entity id, T* data);
    static unsigned bitscan_reverse(std::uint64_t mt);
    static bool find_bitmask_top(
        bitmask_type* bitmask,
        std::uint32_t count,
        std::uint32_t& top_index
    );
    static bool find_bitmask_previous_index(
        bitmask_type* bitmask,
        std::uint32_t index,
        std::uint32_t& prev_index
    );

    struct alignas(T) t_mimicker { std::uint8_t pad[sizeof(T)]; };

    // Bucket data
    std::uint32_t entity_count;
    std::uint32_t bucket_count;
    bitmask_type** bucket_bitmask;
    bitmask_type* top_bitmask;
    entity** bucket_jump_table;
    T** bucket_components;

    // Batching data
    bool batching;
    std::uint32_t batch_checklist_size;
    std::uint32_t batch_checklist_capacity;
    entity* batch_checklist;
    bitmask_type** bucket_batch_bitmask;

    // Search index (kinda separate, but handy to keep around here.)
    scene* ctx;
    search_index<T> search;
};

/** The primary class of the ECS.
 * Entities are created by it, components are attached throught it and events
 * are routed through it.
 */
class scene
{
friend class event_subscription;
public:
    /** The constructor. */
    inline scene();
    /** The destructor.
     * It ensures that all remove_component events are sent for the remainder
     * of the components before event handlers are cleared.
     */
    inline ~scene();

    /** Calls a given function for all suitable entities.
     * The parameters of the function mandate how it is called. Batching is
     * enabled automatically so that removing and adding entities and components
     * during iteration is safe.
     * \param f The iteration callback.
     *   The first parameter of \p f must be #entity (the entity that the
     *   iteration concerns.) After that, further parameters must be either
     *   references or pointers to components. The function is only called when
     *   all referenced components are present for the entity; pointer
     *   parameters are optional and can be null if the component is not
     *   present.
     */
    template<typename F>
    inline void foreach(F&& f);

    /** Same as foreach(), just syntactic sugar.
     * \see foreach()
     */
    template<typename F>
    inline void operator()(F&& f);

    /** Adds an entity without components.
     * \return The new entity ID.
     */
    inline entity add();

    /** Adds an entity with initial components.
     * Takes a list of component instances. Note that they must be moved in, so
     * you should create them during the call or use std::move().
     * \param components All components that should be included.
     * \return The new entity ID.
     */
    template<typename... Components>
    entity add(Components&&... components);

    /** Adds a component to an existing entity, building it in-place.
     * \param id The entity that components are added to.
     * \param args Parameters for the constructor of the Component type.
     */
    template<typename Component, typename... Args>
    void emplace(entity id, Args&&... args);

    /** Adds components to an existing entity.
     * Takes a list of component instances. Note that they must be moved in, so
     * you should create them during the call or use std::move().
     * \param id The entity that components are added to.
     * \param components All components that should be attached.
     */
    template<typename... Components>
    void attach(entity id, Components&&... components);

    /** Removes all components related to the entity.
     * Unlike the component-specific remove() call, this also releases the ID to
     * be reused by another entity.
     * \param id The entity whose components to remove.
     */
    inline void remove(entity id);

    /** Removes a component of an entity.
     * \tparam Component The type of component to remove from the entity.
     * \param id The entity whose component to remove.
     */
    template<typename Component>
    void remove(entity id);

    /** Removes all components of all entities.
     * It also resets the entity counter, so this truly invalidates all
     * previous entities!
     */
    inline void clear_entities();

    /** Copies entities from another ECS to this one.
     * \param other the other ECS whose entities and components to copy to this.
     * \param translation_table if not nullptr, will be filled in with the
     * entity ID correspondence from the old ECS to the new.
     * \warn event handlers are not copied, only entities and components. Entity
     * IDs will also change.
     * \warn You should finish batching on the other ECS before calling this.
     */
    inline void concat(
        scene& other,
        std::map<entity, entity>* translation_table = nullptr
    );

    /** Copies one entity from another ECS to this one.
     * \param other the other ECS whose entity to copy to this.
     * \param other_id the ID of the entity to copy in the other ECS.
     * \return entity ID of the created entity.
     * \warn You should finish batching on the other ECS before calling this.
     */
    inline entity copy(scene& other, entity other_id);

    /** Starts batching behaviour for add/remove.
     * Batching allows you to safely add and remove components while you iterate
     * over them, but comes with no performance benefit.
     */
    inline void start_batch();

    /** Finishes batching behaviour for add/remove and applies the changes.
     */
    inline void finish_batch();

    /** Counts instances of entities with a specified component.
     * \tparam Component the component type to count instances of.
     * \return The number of entities with the specified component.
     */
    template<typename Component>
    size_t count() const;

    /** Checks if an entity has the given component.
     * \tparam Component the component type to check.
     * \param id The id of the entity whose component is checked.
     * \return true if the entity has the given component, false otherwise.
     */
    template<typename Component>
    bool has(entity id) const;

    /** Returns the desired component of an entity.
     * Const version.
     * \tparam Component the component type to get.
     * \param id The id of the entity whose component to fetch.
     * \return A pointer to the component if present, null otherwise.
     */
    template<typename Component>
    const Component* get(entity id) const;

    /** Returns the desired component of an entity.
     * \tparam Component the component type to get.
     * \param id The id of the entity whose component to fetch.
     * \return A pointer to the component if present, null otherwise.
     */
    template<typename Component>
    Component* get(entity id);

    /** Uses search_index<Component> to find the desired component.
     * \tparam Component the component type to search for.
     * \tparam Args search argument types.
     * \param args arguments for search_index<Component>::find().
     * \return A pointer to the component if present, null otherwise.
     * \see update_search_index()
     */
    template<typename Component, typename... Args>
    Component* find_component(Args&&... args);

    /** Uses search_index<Component> to find the desired component.
     * Const version.
     * \tparam Component the component type to search for.
     * \tparam Args search argument types.
     * \param args arguments for search_index<Component>::find().
     * \return A pointer to the component if present, null otherwise.
     * \see update_search_index()
     */
    template<typename Component, typename... Args>
    const Component* find_component(Args&&... args) const;

    /** Uses search_index<Component> to find the desired entity.
     * \tparam Component the component type to search for.
     * \tparam Args search argument types.
     * \param args arguments for search_index<Component>::find().
     * \return An entity id if found, INVALID_ENTITY otherwise.
     * \see update_search_index()
     */
    template<typename Component, typename... Args>
    entity find(Args&&... args) const;

    /** Calls search_index<Component>::update() for the component type.
     * \tparam Component the component type whose search index to update.
     */
    template<typename Component>
    void update_search_index();

    /** Updates search indices of all component types.
     */
    inline void update_search_indices();

    /** Calls all handlers of the given event type.
     * \tparam EventType the type of the event to emit.
     * \param event The event to emit.
     */
    template<typename EventType>
    void emit(const EventType& event);

    /** Returns how many handlers are present for the given event type.
     * \tparam EventType the type of the event to check.
     * \return the number of event handlers for this EventType.
     */
    template<typename EventType>
    size_t get_handler_count() const;

    /** Adds event handler(s) to the ECS.
     * \tparam F Callable types, with signature void(scene& ctx, const EventType& e).
     * \param callbacks The event handler callbacks.
     * \return ID of the "subscription"
     * \see subscribe() for RAII handler lifetime.
     * \see bind_event_handler() for binding to member functions.
     */
    template<typename... F>
    size_t add_event_handler(F&&... callbacks);

    /** Adds member functions of an object as event handler(s) to the ECS.
     * \tparam T Type of the object whose members are being bound.
     * \tparam F Member function types, with signature
     * void(scene& ctx, const EventType& e).
     * \param userdata The class to bind to each callback.
     * \param callbacks The event handler callbacks.
     * \return ID of the "subscription"
     * \see subscribe() for RAII handler lifetime.
     * \see add_event_handler() for free-standing functions.
     */
    template<class T, typename... F>
    size_t bind_event_handler(T* userdata, F&&... callbacks);

    /** Removes event handler(s) from the ECS
     * \param id ID of the "subscription"
     * \see subscribe() for RAII handler lifetime.
     */
    inline void remove_event_handler(size_t id);

    /** Adds event handlers for a receiver object.
     * \tparam EventTypes Event types that are being received.
     * \param r The receiver to add handlers for.
     */
    template<typename... EventTypes>
    void add_receiver(receiver<EventTypes...>& r);

    /** Adds event handlers with a subscription object that tracks lifetime.
     * \tparam F Callable types, with signature void(scene& ctx, const EventType& e).
     * \param callbacks The event handler callbacks.
     * \return The subscription object that removes the event handler on its
     * destruction.
     */
    template<typename... F>
    event_subscription subscribe(F&&... callbacks);

private:
    template<bool pass_id, typename... Components>
    struct foreach_impl
    {
        template<typename F>
        static void foreach(scene& ctx, F&& f);

        template<typename Component>
        struct iterator_wrapper
        {
            static constexpr bool required = true;
            typename component_container<std::decay_t<std::remove_pointer_t<std::decay_t<Component>>>>::iterator iter;
        };

        template<typename Component>
        static inline auto make_iterator(scene& ctx)
        {
            return iterator_wrapper<Component>{
                ctx.get_container<std::decay_t<std::remove_pointer_t<std::decay_t<Component>>>>().begin()
            };
        }

        template<typename Component>
        struct converter
        {
            template<typename T>
            static inline T& convert(T*);
        };

        template<typename F>
        static void call(
            F&& f,
            entity id,
            std::decay_t<std::remove_pointer_t<std::decay_t<Components>>>*... args
        );
    };

    template<typename... Components>
    foreach_impl<true, Components...>
    foreach_redirector(const std::function<void(entity id, Components...)>&);

    template<typename... Components>
    foreach_impl<false, Components...>
    foreach_redirector(const std::function<void(Components...)>&);

    template<typename T>
    T event_handler_type_detector(const std::function<void(scene&, const T&)>&);

    template<typename T>
    T event_handler_type_detector(void (*)(scene&, const T&));

    template<typename T, typename U>
    U event_handler_type_detector(void (T::*)(scene&, const U&));

    template<typename Component>
    void try_attach_dependencies(entity id);

    template<typename Component>
    component_container<Component>& get_container() const;

    template<typename Component>
    static size_t get_component_type_key();
    inline static size_t component_type_key_counter = 0;

    template<typename Event>
    static size_t get_event_type_key();
    inline static size_t event_type_key_counter = 0;

    template<typename F>
    void internal_add_handler(size_t id, F&& f);

    template<class C, typename F>
    void internal_bind_handler(size_t id, C* c, F&& f);

    entity id_counter;
    std::vector<entity> reusable_ids;
    std::vector<entity> post_batch_reusable_ids;
    size_t subscriber_counter;
    int defer_batch;
    mutable std::vector<std::unique_ptr<component_container_base>> components;

    struct event_handler
    {
        size_t subscription_id;

        // TODO: Once we have std::function_ref, see if this can be optimized.
        // The main difficulty we have to handle are pointers to member
        // functions, which have been made unnecessarily unusable in C++.
        std::function<void(scene& ctx, const void* event)> callback;
    };
    std::vector<std::vector<event_handler>> event_handlers;
};

/** Components may derive from this class to require other components.
 * The other components are added to the entity along with this one if they
 * are not yet present.
 */
template<typename... DependencyComponents>
class dependency_components
{
friend class scene;
public:
    static void ensure_dependency_components_exist(entity id, scene& ctx);
};


//==============================================================================
// Implementation
//==============================================================================

event_subscription::event_subscription(scene* ctx, std::size_t subscription_id)
: ctx(ctx), subscription_id(subscription_id)
{
}

event_subscription::event_subscription(event_subscription&& other)
: ctx(other.ctx), subscription_id(other.subscription_id)
{
    other.ctx = nullptr;
}

event_subscription::~event_subscription()
{
    if(ctx)
        ctx->remove_event_handler(subscription_id);
}

template<typename T>
constexpr bool search_index_is_empty_default(
    int,
    typename T::empty_default_impl const * = nullptr
){ return true; }

template<typename T>
constexpr bool search_index_is_empty_default(long)
{ return false; }

template<typename T>
constexpr bool search_index_is_empty_default()
 { return search_index_is_empty_default<T>(0); }

template<typename Component>
void search_index<Component>::add_entity(entity, const Component&) {}

template<typename Component>
void search_index<Component>::update(scene&) {}

template<typename Component>
void search_index<Component>::remove_entity(entity, const Component&) {}

template<typename T>
component_container<T>::component_container(scene& ctx)
:   entity_count(0), bucket_count(0),
    bucket_bitmask(nullptr), top_bitmask(nullptr),
    bucket_jump_table(nullptr), bucket_components(nullptr), batching(false),
    batch_checklist_size(0), batch_checklist_capacity(0),
    batch_checklist(nullptr), bucket_batch_bitmask(nullptr), ctx(&ctx)
{
}

template<typename T>
component_container<T>::~component_container()
{
    destroy();
}

template<typename T>
T* component_container<T>::operator[](entity e)
{
    if(!contains(e)) return nullptr;
    return get_unsafe(e);
}

template<typename T>
const T* component_container<T>::operator[](entity e) const
{
    return const_cast<component_container<T>*>(this)->operator[](e);
}

template<typename T>
void component_container<T>::insert(entity id, T&& value)
{
    emplace(id, std::move(value));
}

template<typename T>
template<typename... Args>
void component_container<T>::emplace(entity id, Args&&... args)
{
    if(id == INVALID_ENTITY)
        return;

    ensure_bucket_space(id);
    if(contains(id))
    { // If we just replace something that exists, life is easy.
        bucket_erase(id, true);
        bucket_insert(id, std::forward<Args>(args)...);
    }
    else if(batching)
    {
        entity_count++;
        if(!batch_change(id))
        {
            // If there was already a change, that means that there was an
            // existing batched erase. That means that we can replace an
            // existing object instead.
            bucket_erase(id, true);
        }
        bucket_insert(id, std::forward<Args>(args)...);
    }
    else
    {
        entity_count++;
        bitmask_insert(id);
        jump_table_insert(id);
        bucket_insert(id, std::forward<Args>(args)...);
    }
}

template<typename T>
void component_container<T>::erase(entity id)
{
    if(!contains(id))
        return;
    entity_count--;

    if(batching)
    {
        if(!batch_change(id))
        {
            // If there was already a change, that means that there was an
            // existing batched add. Because it's not being iterated, we can
            // just destroy it here.
            bucket_erase(id, true);
        }
        else signal_remove(id, get_unsafe(id));
    }
    else
    {
        bool erase_bucket = bitmask_erase(id);
        jump_table_erase(id);
        bucket_erase(id, true);
        if(erase_bucket)
        {
            bucket_self_erase(id >> bucket_exp);
            try_jump_table_bucket_erase(id >> bucket_exp);
        }
    }
}

template<typename T>
void component_container<T>::clear()
{
    if(batching)
    { // Uh oh, this is super suboptimal :/ pls don't clear while iterating.
        for(auto it = begin(); it != end(); ++it)
        {
            erase((*it).first);
        }
    }
    else
    {
        // Clear top bitmask
        std::uint32_t top_bitmask_count = get_top_bitmask_size();
        for(std::uint32_t i = 0; i< top_bitmask_count; ++i)
            top_bitmask[i] = 0;

        // Destroy all existing objects
        if(
            ctx->get_handler_count<remove_component<T>>() ||
            !search_index_is_empty_default<decltype(search)>()
        ){
            for(auto it = begin(); it != end(); ++it)
            {
                auto pair = *it;
                signal_remove(pair.first, pair.second);
                pair.second->~T();
            }
        }
        else
        {
            for(auto it = begin(); it != end(); ++it)
                (*it).second->~T();
        }

        // Release all bucket pointers
        for(std::uint32_t i = 0; i < bucket_count; ++i)
        {
            if(bucket_bitmask[i])
            {
                delete [] bucket_bitmask[i];
                bucket_bitmask[i] = nullptr;
            }
            if(bucket_batch_bitmask[i])
            {
                delete [] bucket_batch_bitmask[i];
                bucket_batch_bitmask[i] = nullptr;
            }
            if(bucket_jump_table[i])
            {
                delete [] bucket_jump_table[i];
                bucket_jump_table[i] = nullptr;
            }
            if constexpr(!tag_component)
            {
                if(bucket_components[i])
                {
                    delete [] reinterpret_cast<t_mimicker*>(bucket_components[i]);
                    bucket_components[i] = nullptr;
                }
            }
        }
    }
    entity_count = 0;
}

template<typename T>
bool component_container<T>::contains(entity id) const
{
    entity hi = id >> bucket_exp;
    if(id == INVALID_ENTITY || hi >= bucket_count) return false;
    entity lo = id & bucket_mask;
    if(batching)
    {
        bitmask_type bitmask = bucket_bitmask[hi] ?
            bucket_bitmask[hi][lo>>bitmask_shift] : 0;
        if(bucket_batch_bitmask[hi])
            bitmask ^= bucket_batch_bitmask[hi][lo>>bitmask_shift];
        return (bitmask >> (lo&bitmask_mask))&1;
    }
    else
    {
        bitmask_type* bitmask = bucket_bitmask[hi];
        if(!bitmask) return false;
        return (bitmask[lo>>bitmask_shift] >> (lo&bitmask_mask))&1;
    }
}

template<typename T>
void component_container<T>::start_batch()
{
    batching = true;
    batch_checklist_size = 0;
}

template<typename T>
void component_container<T>::finish_batch()
{
    if(!batching) return;
    batching = false;

    // Discard duplicate changes first.
    for(std::uint32_t i = 0; i < batch_checklist_size; ++i)
    {
        std::uint32_t ri = batch_checklist_size-1-i;
        entity& id = batch_checklist[ri];
        entity hi = id >> bucket_exp;
        entity lo = id & bucket_mask;
        bitmask_type bit = std::uint64_t(1)<<(lo&bitmask_mask);
        bitmask_type* bbit = bucket_batch_bitmask[hi];
        if(bbit && (bbit[lo>>bitmask_shift] & bit))
        { // Not a dupe, but latest state.
            bbit[lo>>bitmask_shift] ^= bit;
        }
        else id = INVALID_ENTITY;
    }

    // Now, do all changes for realzies. All IDs that are left are unique and
    // change the existence of an entity.
    for(std::uint32_t i = 0; i < batch_checklist_size; ++i)
    {
        entity& id = batch_checklist[i];
        if(id == INVALID_ENTITY) continue;

        entity hi = id >> bucket_exp;
        entity lo = id & bucket_mask;
        bitmask_type bit = std::uint64_t(1)<<(lo&bitmask_mask);
        if(bucket_bitmask[hi] && (bucket_bitmask[hi][lo>>bitmask_shift] & bit))
        { // Erase
            bitmask_erase(id);
            jump_table_erase(id);
            bucket_erase(id, false);
        }
        else
        { // Insert (in-place)
            bitmask_insert(id);
            jump_table_insert(id);
            // No need to add to bucket, that already happened due to batching
            // semantics.
        }
    }

    // Finally, check erased entries for if we can remove their buckets.
    for(std::uint32_t i = 0; i < batch_checklist_size; ++i)
    {
        entity& id = batch_checklist[i];
        if(id == INVALID_ENTITY) continue;

        entity hi = id >> bucket_exp;
        if(bucket_bitmask[hi] == nullptr) // Already erased!
            continue;

        entity lo = id & bucket_mask;
        if(bucket_bitmask[hi][lo>>bitmask_shift] == 0 && bitmask_empty(hi))
        { // This got erased, so check if the whole bucket is empty.
            bucket_self_erase(hi);
            try_jump_table_bucket_erase(id >> bucket_exp);
        }
    }
}

template<typename T>
typename component_container<T>::iterator component_container<T>::begin()
{
    if(entity_count == 0) return end();
    // The jump entry for INVALID_ENTITY stores the first valid entity index.
    // INVALID_ENTITY is always present, but doesn't cause allocation of a
    // component for itself.
    return iterator(*this, bucket_jump_table[0][0]);
}

template<typename T>
typename component_container<T>::iterator component_container<T>::end()
{
    return iterator(*this, INVALID_ENTITY);
}

template<typename T>
std::size_t component_container<T>::size() const
{
    return entity_count;
}

template<typename T>
void component_container<T>::update_search_index()
{
    search.update(*ctx);
}

template<typename T>
void component_container<T>::list_entities(
    std::map<entity, entity>& translation_table
){
    for(auto it = begin(); it; ++it)
        translation_table[(*it).first] = INVALID_ENTITY;
}

template<typename T>
void component_container<T>::concat(
    scene& target,
    const std::map<entity, entity>& translation_table
){
    if constexpr(std::is_copy_constructible_v<T>)
    {
        for(auto it = begin(); it; ++it)
        {
            auto pair = *it;
            target.emplace<T>(translation_table.at(pair.first), *pair.second);
        }
    }
}

template<typename T>
void component_container<T>::copy(
    scene& target,
    entity result_id,
    entity original_id
){
    if constexpr(std::is_copy_constructible_v<T>)
    {
        T* comp = operator[](original_id);
        if(comp) target.emplace<T>(result_id, *comp);
    }
}

template<typename T>
template<typename... Args>
entity component_container<T>::find_entity(Args&&... args) const
{
    return search.find(std::forward<Args>(args)...);
}

template<typename T>
T* component_container<T>::get_unsafe(entity e)
{
    if constexpr(tag_component)
    {
        // We can return basically anything, since tag components are just tags.
        // As long as it's not nullptr, that is.
        return reinterpret_cast<T*>(&bucket_components);
    }
    else return &bucket_components[e>>bucket_exp][e&bucket_mask];
}

template<typename T>
void component_container<T>::destroy()
{
    // Cannot batch while destroying.
    if(batching)
        finish_batch();

    // Destruct all entries
    clear();
#ifndef MONKERO_CONTAINER_DEALLOCATE_BUCKETS
    for(size_t i = 0; i < bucket_count; ++i)
    {
        if(bucket_bitmask[i])
            delete bucket_bitmask[i];
        if(bucket_jump_table[i])
            delete bucket_jump_table[i];
        if constexpr(!tag_component)
        {
            if(bucket_components[i])
                delete bucket_components[i];
        }
        if(bucket_batch_bitmask[i])
            delete bucket_batch_bitmask[i];
    }
#endif

    // Free top-level arrays (bottom-level arrays should have been deleted in
    // clear().
    if(bucket_bitmask)
        delete[] bucket_bitmask;
    if(top_bitmask)
        delete[] top_bitmask;
    if(bucket_jump_table)
    {
        // The initial jump table entry won't get erased otherwise, as it is a
        // special case due to INVALID_ENTITY.
        delete[] bucket_jump_table[0];
        delete[] bucket_jump_table;
    }
    if constexpr(!tag_component)
    {
        if(bucket_components)
            delete[] bucket_components;
    }
    if(batch_checklist)
        delete[] batch_checklist;
    if(bucket_batch_bitmask)
        delete[] bucket_batch_bitmask;
}

template<typename T>
void component_container<T>::jump_table_insert(entity id)
{
    // Assumes that the corresponding bitmask change has already been made.
    std::uint32_t cur_hi = id >> bucket_exp;
    std::uint32_t cur_lo = id & bucket_mask;
    ensure_jump_table(cur_hi);

    // Find the start of the preceding block.
    entity prev_start_id = find_previous_entity(id);
    std::uint32_t prev_start_hi = prev_start_id >> bucket_exp;
    std::uint32_t prev_start_lo = prev_start_id & bucket_mask;
    ensure_jump_table(prev_start_hi);
    entity& prev_start = bucket_jump_table[prev_start_hi][prev_start_lo];

    if(prev_start_id + 1 < id)
    { // Make preceding block's end point back to its start
        entity prev_end_id = id-1;
        std::uint32_t prev_end_hi = prev_end_id >> bucket_exp;
        std::uint32_t prev_end_lo = prev_end_id & bucket_mask;
        ensure_jump_table(prev_end_hi);
        entity& prev_end = bucket_jump_table[prev_end_hi][prev_end_lo];
        prev_end = prev_start_id;
    }

    if(id + 1 < prev_start)
    { // Make succeeding block's end point back to its start
        entity next_end_id = prev_start-1;
        std::uint32_t next_end_hi = next_end_id >> bucket_exp;
        std::uint32_t next_end_lo = next_end_id & bucket_mask;
        entity& next_end = bucket_jump_table[next_end_hi][next_end_lo];
        next_end = id;
    }

    bucket_jump_table[cur_hi][cur_lo] = prev_start;
    prev_start = id;
}

template<typename T>
void component_container<T>::jump_table_erase(entity id)
{
    std::uint32_t hi = id >> bucket_exp;
    std::uint32_t lo = id & bucket_mask;
    entity prev = id-1;
    std::uint32_t prev_hi = prev >> bucket_exp;
    std::uint32_t prev_lo = prev & bucket_mask;

    entity& prev_jmp = bucket_jump_table[prev_hi][prev_lo];
    entity& cur_jmp = bucket_jump_table[hi][lo];
    entity block_start = 0;
    if(prev_jmp == id)
    { // If previous existed
        // It should jump where the current entity would have jumped
        prev_jmp = cur_jmp;
        block_start = prev;
    }
    else
    { // If previous did not exist
        // Find the starting entry of this block from it.
        prev_hi = prev_jmp >> bucket_exp;
        prev_lo = prev_jmp & bucket_mask;
        // Update the starting entry to jump to our target.
        bucket_jump_table[prev_hi][prev_lo] = cur_jmp;
        block_start = prev_jmp;
    }

    // Ensure the skip block end knows to jump back as well.
    if(cur_jmp != 0)
    {
        entity block_end = cur_jmp-1;
        prev_hi = block_end >> bucket_exp;
        prev_lo = block_end & bucket_mask;
        bucket_jump_table[prev_hi][prev_lo] = block_start;
    }
}

template<typename T>
std::size_t component_container<T>::get_top_bitmask_size() const
{
    return bucket_count == 0 ? 0 : std::max(initial_bucket_count, bucket_count >> bitmask_shift);
}

template<typename T>
bool component_container<T>::bitmask_empty(std::uint32_t bucket_index) const
{
    if(bucket_bitmask[bucket_index] == nullptr)
        return true;
    for(unsigned j = 0; j < bucket_bitmask_units; ++j)
    {
        if(bucket_bitmask[bucket_index][j] != 0)
            return false;
    }
    return true;
}

template<typename T>
void component_container<T>::bitmask_insert(entity id)
{
    std::uint32_t hi = id >> bucket_exp;
    std::uint32_t lo = id & bucket_mask;
    ensure_bitmask(hi);
    bitmask_type& mask = bucket_bitmask[hi][lo>>bitmask_shift];
    if(mask == 0)
        top_bitmask[hi>>bitmask_shift] |= std::uint64_t(1)<<(hi&bitmask_mask);
    mask |= std::uint64_t(1)<<(lo&bitmask_mask);
}

template<typename T>
bool component_container<T>::bitmask_erase(entity id)
{
    std::uint32_t hi = id >> bucket_exp;
    std::uint32_t lo = id & bucket_mask;
    bucket_bitmask[hi][lo>>bitmask_shift] &= ~(std::uint64_t(1)<<(lo&bitmask_mask));
    if(bucket_bitmask[hi][lo>>bitmask_shift] == 0 && bitmask_empty(hi))
    {
        top_bitmask[hi>>bitmask_shift] &= ~(std::uint64_t(1)<<(hi&bitmask_mask));
        return true;
    }
    return false;
}

template<typename T>
template<typename... Args>
void component_container<T>::bucket_insert(entity id, Args&&... args)
{
    // This function assumes that there isn't an existing entity at the same
    // position.
    T* data = nullptr;
    if constexpr(tag_component)
    {
        data = reinterpret_cast<T*>(&bucket_components);
        new (&bucket_components) T(std::forward<Args>(args)...);
    }
    else
    {
        std::uint32_t hi = id >> bucket_exp;
        std::uint32_t lo = id & bucket_mask;

        // If this component container doesn't exist yet, create it.
        if(bucket_components[hi] == nullptr)
        {
            bucket_components[hi] = reinterpret_cast<T*>(
                new t_mimicker[1u<<bucket_exp]
            );
        }
        data = &bucket_components[hi][lo];
    }
    // Create the related component here.
    new (data) T(std::forward<Args>(args)...);
    signal_add(id, data);
}

template<typename T>
void component_container<T>::bucket_erase(entity id, bool signal)
{
    // This function assumes that the given entity exists.
    T* data = nullptr;
    if constexpr(tag_component)
    {
        data = reinterpret_cast<T*>(&bucket_components);
    }
    else
    {
        std::uint32_t hi = id >> bucket_exp;
        std::uint32_t lo = id & bucket_mask;
        data = &bucket_components[hi][lo];
    }
    if(signal) signal_remove(id, data);
    data->~T();
}

template<typename T>
void component_container<T>::bucket_self_erase(std::uint32_t i)
{
    (void)i;
#ifdef MONKERO_CONTAINER_DEALLOCATE_BUCKETS
    // If the bucket got emptied, nuke it.
    delete[] bucket_bitmask[i];
    bucket_bitmask[i] = nullptr;

    if(bucket_batch_bitmask[i])
    {
        delete[] bucket_batch_bitmask[i];
        bucket_batch_bitmask[i] = nullptr;
    }

    if constexpr(!tag_component)
    {
        delete[] reinterpret_cast<t_mimicker*>(bucket_components[i]);
        bucket_components[i] = nullptr;
    }
#endif
}

template<typename T>
void component_container<T>::try_jump_table_bucket_erase(std::uint32_t i)
{
    (void)i;
#ifdef MONKERO_CONTAINER_DEALLOCATE_BUCKETS
    // The first jump table bucket is always present, due to INVALID_ENTITY
    // being the iteration starting point.
    if(i == 0 || bucket_jump_table[i] == nullptr)
        return;

    // We can be removed if the succeeding bucket is also empty.
    if(i+1 >= bucket_count || bucket_bitmask[i+1] == nullptr)
    {
        delete[] bucket_jump_table[i];
        bucket_jump_table[i] = nullptr;
    }
#endif
}

template<typename T>
void component_container<T>::ensure_bucket_space(entity id)
{
    if((id>>bucket_exp) < bucket_count)
        return;

    std::uint32_t new_bucket_count = std::max(initial_bucket_count, bucket_count);
    while(new_bucket_count <= (id>>bucket_exp))
        new_bucket_count *= 2;

    bitmask_type** new_bucket_batch_bitmask = new bitmask_type*[new_bucket_count];
    memcpy(new_bucket_batch_bitmask, bucket_batch_bitmask,
        sizeof(bitmask_type*)*bucket_count);
    memset(new_bucket_batch_bitmask+bucket_count, 0,
        sizeof(bitmask_type*)*(new_bucket_count-bucket_count));
    delete [] bucket_batch_bitmask;
    bucket_batch_bitmask = new_bucket_batch_bitmask;

    bitmask_type** new_bucket_bitmask = new bitmask_type*[new_bucket_count];
    memcpy(new_bucket_bitmask, bucket_bitmask,
        sizeof(bitmask_type*)*bucket_count);
    memset(new_bucket_bitmask+bucket_count, 0,
        sizeof(bitmask_type*)*(new_bucket_count-bucket_count));
    delete [] bucket_bitmask;
    bucket_bitmask = new_bucket_bitmask;

    entity** new_bucket_jump_table = new entity*[new_bucket_count];
    memcpy(new_bucket_jump_table, bucket_jump_table,
        sizeof(entity*)*bucket_count);
    memset(new_bucket_jump_table+bucket_count, 0,
        sizeof(entity*)*(new_bucket_count-bucket_count));
    delete [] bucket_jump_table;
    bucket_jump_table = new_bucket_jump_table;

    // Create initial jump table entry.
    if(bucket_count == 0)
    {
        bucket_jump_table[0] = new entity[1 << bucket_exp];
        memset(bucket_jump_table[0], 0, sizeof(entity)*(1 << bucket_exp));
    }

    if constexpr(!tag_component)
    {
        T** new_bucket_components = new T*[new_bucket_count];
        memcpy(new_bucket_components, bucket_components,
            sizeof(T*)*bucket_count);
        memset(new_bucket_components+bucket_count, 0,
            sizeof(T*)*(new_bucket_count-bucket_count));
        delete [] bucket_components;
        bucket_components = new_bucket_components;
    }

    std::uint32_t top_bitmask_count = get_top_bitmask_size();
    std::uint32_t new_top_bitmask_count = std::max(
        initial_bucket_count,
        new_bucket_count >> bitmask_shift
    );
    if(top_bitmask_count != new_top_bitmask_count)
    {
        bitmask_type* new_top_bitmask = new bitmask_type[new_top_bitmask_count];
        memcpy(new_top_bitmask, top_bitmask,
            sizeof(bitmask_type)*top_bitmask_count);
        memset(new_top_bitmask+top_bitmask_count, 0,
            sizeof(bitmask_type)*(new_top_bitmask_count-top_bitmask_count));
        delete [] top_bitmask;
        top_bitmask = new_top_bitmask;
    }

    bucket_count = new_bucket_count;
}

template<typename T>
void component_container<T>::ensure_bitmask(std::uint32_t bucket_index)
{
    if(bucket_bitmask[bucket_index] == nullptr)
    {
        bucket_bitmask[bucket_index] = new bitmask_type[bucket_bitmask_units];
        std::memset(
            bucket_bitmask[bucket_index], 0,
            sizeof(bitmask_type)*bucket_bitmask_units
        );
    }
}

template<typename T>
void component_container<T>::ensure_jump_table(std::uint32_t bucket_index)
{
    if(!bucket_jump_table[bucket_index])
    {
        bucket_jump_table[bucket_index] = new entity[1 << bucket_exp];
        memset(bucket_jump_table[bucket_index], 0, sizeof(entity)*(1 << bucket_exp));
    }
}

template<typename T>
bool component_container<T>::batch_change(entity id)
{
    std::uint32_t hi = id >> bucket_exp;
    std::uint32_t lo = id & bucket_mask;
    if(bucket_batch_bitmask[hi] == nullptr)
    {
        bucket_batch_bitmask[hi] = new bitmask_type[bucket_bitmask_units];
        std::memset(
            bucket_batch_bitmask[hi], 0,
            sizeof(bitmask_type)*bucket_bitmask_units
        );
    }
    bitmask_type& mask = bucket_batch_bitmask[hi][lo>>bitmask_shift];
    bitmask_type bit = std::uint64_t(1)<<(lo&bitmask_mask);
    mask ^= bit;
    if(mask & bit)
    { // If there will be a change, add this to the list.
        if(batch_checklist_size == batch_checklist_capacity)
        {
            std::uint32_t new_batch_checklist_capacity = std::max(
                initial_bucket_count,
                batch_checklist_capacity * 2
            );
            entity* new_batch_checklist = new entity[new_batch_checklist_capacity];
            memcpy(new_batch_checklist, batch_checklist,
                sizeof(entity)*batch_checklist_capacity);
            memset(new_batch_checklist + batch_checklist_capacity, 0,
                sizeof(entity)*(new_batch_checklist_capacity-batch_checklist_capacity));
            delete [] batch_checklist;
            batch_checklist = new_batch_checklist;
            batch_checklist_capacity = new_batch_checklist_capacity;
        }
        batch_checklist[batch_checklist_size] = id;
        batch_checklist_size++;
        return true;
    }
    return false;
}

template<typename T>
entity component_container<T>::find_previous_entity(entity id)
{
    std::uint32_t hi = id >> bucket_exp;
    std::uint32_t lo = id & bucket_mask;

    // Try to find in the current bucket.
    std::uint32_t prev_index = 0;
    if(find_bitmask_previous_index(bucket_bitmask[hi], lo, prev_index))
        return (hi << bucket_exp) + prev_index;

    // If that failed, search from the top bitmask.
    std::uint32_t bucket_index = 0;
    if(!find_bitmask_previous_index(top_bitmask, hi, bucket_index))
        return INVALID_ENTITY;

    // Now, find the highest bit in the bucket that was found.
    find_bitmask_top(
        bucket_bitmask[bucket_index],
        bucket_bitmask_units,
        prev_index
    );
    return (bucket_index << bucket_exp) + prev_index;
}


template<typename T>
void component_container<T>::signal_add(entity id, T* data)
{
    search.add_entity(id, *data);
    ctx->emit(add_component<T>{id, data});
}

template<typename T>
void component_container<T>::signal_remove(entity id, T* data)
{
    search.remove_entity(id, *data);
    ctx->emit(remove_component<T>{id, data});
}

template<typename T>
unsigned component_container<T>::bitscan_reverse(std::uint64_t mt)
{
#if defined(__GNUC__)
    return 63 - __builtin_clzll(mt);
#elif defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanReverse64(&index, mt);
    return index;
#else
    unsigned r = (mt > 0xFFFFFFFFF) << 5;
    mt >>= r;
    unsigned shift = (mt > 0xFFFF) << 4;
    mt >>= shift;
    r |= shift;
    shift = (mt > 0xFF) << 3;
    mt >>= shift;
    r |= shift;
    shift = (mt > 0xF) << 2;
    mt >>= shift;
    r |= shift;
    shift = (mt > 0x3) << 1;
    mt >>= shift;
    r |= shift;
    return r | (mt >> 1);
#endif
}

template<typename T>
bool component_container<T>::find_bitmask_top(
    bitmask_type* bitmask,
    std::uint32_t count,
    std::uint32_t& top_index
){
    for(std::uint32_t j = 0, i = count-1; j < count; ++j, --i)
    {
        if(bitmask[i] != 0)
        {
            std::uint32_t index = bitscan_reverse(bitmask[i]);
            top_index = (i << bitmask_shift) + index;
            return true;
        }
    }

    return false;
}

template<typename T>
bool component_container<T>::find_bitmask_previous_index(
    bitmask_type* bitmask,
    std::uint32_t index,
    std::uint32_t& prev_index
){
    if(!bitmask)
        return false;

    std::uint32_t bm_index = index >> bitmask_shift;
    bitmask_type bm_mask = (std::uint64_t(1)<<(index&bitmask_mask))-1;
    bitmask_type cur_mask = bitmask[bm_index] & bm_mask;
    if(cur_mask != 0)
    {
        std::uint32_t index = bitscan_reverse(cur_mask);
        prev_index = (bm_index << bitmask_shift) + index;
        return true;
    }

    return find_bitmask_top(bitmask, bm_index, prev_index);
}

template<typename T>
component_container<T>::iterator::iterator(component_container& from, entity e)
:   from(&from), current_entity(e), current_bucket(e>>bucket_exp)
{
    if(current_bucket < from.bucket_count)
    {
        current_jump_table = from.bucket_jump_table[current_bucket];
        if constexpr(!tag_component)
        {
            current_components = from.bucket_components[current_bucket];
        }
    }
}

void component_container_entity_advancer::advance()
{
    current_entity = current_jump_table[current_entity&bucket_mask];
    std::uint32_t next_bucket = current_entity >> bucket_exp;
    if(next_bucket != current_bucket)
    {
        current_bucket = next_bucket;
        current_jump_table = (*bucket_jump_table)[current_bucket];
    }
}

template<typename T>
typename component_container<T>::iterator& component_container<T>::iterator::operator++()
{
    current_entity = current_jump_table[current_entity&bucket_mask];
    std::uint32_t next_bucket = current_entity >> bucket_exp;
    if(next_bucket != current_bucket)
    {
        current_bucket = next_bucket;
        current_jump_table = from->bucket_jump_table[current_bucket];
        if constexpr(!tag_component)
        {
            current_components = from->bucket_components[current_bucket];
        }
    }
    return *this;
}

template<typename T>
typename component_container<T>::iterator component_container<T>::iterator::operator++(int)
{
    iterator it(*this);
    ++it;
    return it;
}

template<typename T>
std::pair<entity, T*> component_container<T>::iterator::operator*()
{
    if constexpr(tag_component)
    {
        return {
            current_entity,
            reinterpret_cast<T*>(&from->bucket_components)
        };
    }
    else
    {
        return {
            current_entity,
            &current_components[current_entity&bucket_mask]
        };
    }
}

template<typename T>
std::pair<entity, const T*> component_container<T>::iterator::operator*() const
{
    if constexpr(tag_component)
    {
        return {
            current_entity,
            reinterpret_cast<const T*>(&from->bucket_components)
        };
    }
    else
    {
        return {
            current_entity,
            &current_components[current_entity&bucket_mask]
        };
    }
}

template<typename T>
bool component_container<T>::iterator::operator==(const iterator& other) const
{
    return other.current_entity == current_entity;
}

template<typename T>
bool component_container<T>::iterator::operator!=(const iterator& other) const
{
    return other.current_entity != current_entity;
}

template<typename T>
bool component_container<T>::iterator::try_advance(entity id)
{
    if(current_entity == id)
        return true;

    std::uint32_t next_bucket = id >> bucket_exp;
    std::uint32_t lo = id & bucket_mask;
    if(
        id < current_entity ||
        next_bucket >= from->bucket_count ||
        !from->bucket_bitmask[next_bucket] ||
        !(from->bucket_bitmask[next_bucket][lo>>bitmask_shift] & (std::uint64_t(1) << (lo&bitmask_mask)))
    ) return false;

    current_entity = id;
    if(next_bucket != current_bucket)
    {
        current_bucket = next_bucket;
        current_jump_table = from->bucket_jump_table[current_bucket];
        if constexpr(!tag_component)
            current_components = from->bucket_components[current_bucket];
    }
    return true;
}

template<typename T>
component_container<T>::iterator::operator bool() const
{
    return current_entity != INVALID_ENTITY;
}

template<typename T>
entity component_container<T>::iterator::get_id() const
{
    return current_entity;
}

template<typename T>
component_container<T>* component_container<T>::iterator::get_container() const
{
    return from;
}

template<typename T>
component_container_entity_advancer component_container<T>::iterator::get_advancer()
{
    return component_container_entity_advancer{
        bucket_mask,
        bucket_exp,
        &from->bucket_jump_table,
        current_bucket,
        current_entity,
        current_jump_table
    };
}

#ifdef MONKERO_CONTAINER_DEBUG_UTILS
template<typename T>
bool component_container<T>::test_invariant() const
{
    // Check bitmask internal validity
    std::uint32_t top_bitmask_count = get_top_bitmask_size();
    std::uint32_t top_index = 0;
    bool found = top_bitmask && find_bitmask_top(
        top_bitmask,
        top_bitmask_count,
        top_index
    );
    std::uint32_t bitmask_entity_count = 0;
    if(found && top_index >= bucket_count && !batching)
    {
        std::cout << "Top bitmask has a higher bit than bucket count!\n";
        return false;
    }

    for(std::uint32_t i = 0; i < bucket_count; ++i)
    {
        int present = (top_bitmask[i>>bitmask_shift] >> (i&bitmask_mask))&1;
        if(present && !bucket_bitmask[i] && !batching)
        {
            std::cout << "Bitmask bucket that should exist is null instead!\n";
            return false;
        }
        bool found = bucket_bitmask[i] && find_bitmask_top(
            bucket_bitmask[i],
            bucket_bitmask_units,
            top_index
        );
        if(present && !found && !batching)
        {
            std::cout << "Empty bitmask bucket marked as existing in the top-level!\n";
            return false;
        }
        else if(!present && found && !batching)
        {
            std::cout << "Non-empty bitmask bucket marked as nonexistent in the top-level!\n";
            return false;
        }
        if(bucket_bitmask[i])
        {
            for(std::uint32_t j = 0; j < bucket_bitmask_units; ++j)
            {
                bitmask_entity_count += __builtin_popcountll(bucket_bitmask[i][j]);
            }
        }
    }

    if(!batching && bitmask_entity_count != entity_count)
    {
        std::cout << "Number of entities in bitmask does not match tracked number!\n";
        return false;
    }

    // Check jump table internal validity
    std::uint32_t jump_table_entity_count = 0;
    if(entity_count != 0)
    {
        entity prev_id = 0;
        entity id = bucket_jump_table[0][0];
        while(id != 0)
        {
            bitmask_type* bm = bucket_bitmask[id>>bucket_exp];
            bool present = false;
            if(bm)
            {
                entity lo = id & bucket_mask;
                present = (bm[lo>>bitmask_shift] >> (lo&bitmask_mask))&1;
            }
            if(!present && !batching)
            {
                std::cout << "Jump table went to a non-existent entity!\n";
                return false;
            }

            entity preceding_id = id-1;
            entity prec_next_id = bucket_jump_table[preceding_id>>bucket_exp][preceding_id&bucket_mask];
            if(prec_next_id != id && prec_next_id != prev_id)
            {
                std::cout << "Jump table preceding entry has invalid target id!\n";
                return false;
            }

            jump_table_entity_count++;
            prev_id = id;
            entity next_id = bucket_jump_table[id>>bucket_exp][id&bucket_mask];
            if(next_id != 0 && next_id <= id)
            {
                std::cout << "Jump table did not jump forward!\n";
                return false;
            }
            id = next_id;
        }
    }

    if(jump_table_entity_count != entity_count && !batching)
    {
        std::cout << "Number of entities in jump table does not match tracked number!\n";
        return false;
    }
    return true;
}

template<typename T>
void component_container<T>::print_bitmask() const
{
    for(std::uint32_t i = 0; i < bucket_count; ++i)
    {
        int present = (top_bitmask[i>>bitmask_shift] >> (i&bitmask_mask))&1;
        std::cout << "bucket " << i << " ("<< (present ? "present" : "empty") << "): ";
        if(!bucket_bitmask[i])
            std::cout << "(null)\n";
        else
        {
            for(std::uint32_t j = 0; j < bucket_bitmask_units; ++j)
            {
                for(std::uint32_t k = 0; k < 64; ++k)
                {
                    std::cout << ((bucket_bitmask[i][j]>>k)&1);
                }
                std::cout << " ";
            }
            std::cout << "\n";
        }
    }
}

template<typename T>
void component_container<T>::print_jump_table() const
{
    std::uint32_t k = 0;
    for(std::uint32_t i = 0; i < bucket_count; ++i)
    {
        if(bucket_jump_table[i] == nullptr)
        {
            std::uint32_t k_start = k;
            std::uint32_t i_start = i;
            for(; i < bucket_count && bucket_jump_table[i] == nullptr; ++i)
                k += 1<<bucket_exp;
            --i;
            std::uint32_t k_end = k-1;
            std::uint32_t i_end = i;
            if(i_start == i_end)
                std::cout << "bucket " << i_start << ": " << k_start << " to " << k_end;
            else
                std::cout << "buckets " << i_start << " to " << i_end << ": " << k_start << " to " << k_end;
        }
        else
        {
            std::cout << "bucket " << i << ":\n";

            std::cout << "\tindices: |";

            for(int j = 0; j < (1<<bucket_exp); ++j, ++k)
                std::cout << " " << k << " |";

            std::cout << "\n\tdata:    |";
            for(int j = 0; j < (1<<bucket_exp); ++j)
            {
                std::cout << " " << bucket_jump_table[i][j] << " |";
            }
        }
        std::cout << "\n";
    }
}
#endif

scene::scene()
: id_counter(1), subscriber_counter(0), defer_batch(0)
{
}

scene::~scene()
{
    // This is called manually so that remove events are fired if necessary.
    clear_entities();
}

template<bool pass_id, typename... Components>
template<typename Component>
struct scene::foreach_impl<pass_id, Components...>::iterator_wrapper<Component*>
{
    static constexpr bool required = false;
    typename component_container<std::decay_t<std::remove_pointer_t<std::decay_t<Component>>>>::iterator iter;
};

template<bool pass_id, typename... Components>
template<typename F>
void scene::foreach_impl<pass_id, Components...>::foreach(scene& ctx, F&& f)
{
    ctx.start_batch();

    std::tuple component_it(make_iterator<Components>(ctx)...);
#define monkero_apply_tuple(...) \
    std::apply([&](auto&... it){return (__VA_ARGS__);}, component_it)

    // Note that all checks based on it.required are compile-time, it's
    // constexpr!
    constexpr bool all_optional = (std::is_pointer_v<Components> && ...);

    if constexpr(sizeof...(Components) == 1)
    {
        // If we're only iterating one category, we can do it very quickly!
        auto& it = std::get<0>(component_it).iter;
        while(it)
        {
            auto [cur_id, ptr] = *it;
            call(std::forward<F>(f), cur_id, ptr);
            ++it;
        }
    }
    else if constexpr(all_optional)
    {
        // If all are optional, iteration logic has to differ a bit. The other
        // version would never quit as there would be zero finished required
        // iterators.
        while(monkero_apply_tuple((bool)it.iter || ...))
        {
            entity cur_id = monkero_apply_tuple(std::min({
                (it.iter ? it.iter.get_id() : std::numeric_limits<entity>::max())...
            }));
            monkero_apply_tuple(call(
                std::forward<F>(f),
                cur_id,
                (it.iter.get_id() == cur_id ? (*it.iter).second : nullptr)...
            ));
            monkero_apply_tuple(
                (it.iter && it.iter.get_id() == cur_id ? (++it.iter, void()) : void()), ...
            );
        }
    }
    else
    {
        // This is the generic implementation for when there's multiple
        // components where some are potentially optional.
        std::size_t min_length = monkero_apply_tuple(std::min({
            (it.required ?
                it.iter.get_container()->size() :
                std::numeric_limits<std::size_t>::max()
            )...
        }));

        component_container_entity_advancer advancer = {};
        monkero_apply_tuple(
            (it.required && it.iter.get_container()->size() == min_length ?
                (advancer = it.iter.get_advancer(), void()): void()), ...
        );

        while(advancer.current_entity != INVALID_ENTITY)
        {
            bool have_all_required = monkero_apply_tuple(
                (it.iter.try_advance(advancer.current_entity) || !it.required) && ...
            );
            if(have_all_required)
            {
                monkero_apply_tuple(call(
                    std::forward<F>(f), advancer.current_entity,
                    (it.iter.get_id() == advancer.current_entity ? (*it.iter).second : nullptr)...
                ));
            }
            advancer.advance();
        }
    }
#undef monkero_apply_tuple

    ctx.finish_batch();
}

template<bool pass_id, typename... Components>
template<typename Component>
struct scene::foreach_impl<pass_id, Components...>::converter<Component*>
{
    template<typename T>
    static inline T* convert(T* val) { return val; }
};

template<bool pass_id, typename... Components>
template<typename Component>
template<typename T>
T& scene::foreach_impl<pass_id, Components...>::converter<Component>::convert(T* val)
{
    return *val;
}

template<bool pass_id, typename... Components>
template<typename F>
void scene::foreach_impl<pass_id, Components...>::call(
    F&& f,
    entity id,
    std::decay_t<std::remove_pointer_t<std::decay_t<Components>>>*... args
){
    if constexpr(pass_id) f(id, converter<Components>::convert(args)...);
    else f(converter<Components>::convert(args)...);
}

template<typename T, typename=void>
struct has_ensure_dependency_components_exist: std::false_type { };

template<typename T>
struct has_ensure_dependency_components_exist<
    T,
    decltype((void)
        T::ensure_dependency_components_exist(entity(), *(scene*)nullptr), void()
    )
> : std::true_type { };

template<typename Component>
void scene::try_attach_dependencies(entity id)
{
    (void)id;
    if constexpr(has_ensure_dependency_components_exist<Component>::value)
        Component::ensure_dependency_components_exist(id, *this);
}

template<typename F>
void scene::foreach(F&& f)
{
    // This one little trick lets us know the argument types without
    // actually using the std::function wrapper at runtime!
    decltype(
        foreach_redirector(std::function(f))
    )::foreach(*this, std::forward<F>(f));
}

template<typename F>
void scene::operator()(F&& f)
{
    foreach(std::forward<F>(f));
}

entity scene::add()
{
    if(reusable_ids.size() > 0)
    {
        entity id = reusable_ids.back();
        reusable_ids.pop_back();
        return id;
    }
    else
    {
        if(id_counter == INVALID_ENTITY)
            return INVALID_ENTITY;
        return id_counter++;
    }
}

template<typename... Components>
entity scene::add(Components&&... components)
{
    entity id = add();
    attach(id, std::forward<Components>(components)...);
    return id;
}

template<typename Component, typename... Args>
void scene::emplace(entity id, Args&&... args)
{
    try_attach_dependencies<Component>(id);

    get_container<Component>().emplace(
        id, std::forward<Args>(args)...
    );
}

template<typename... Components>
void scene::attach(entity id, Components&&... components)
{
    (try_attach_dependencies<Components>(id), ...);

    (
        get_container<Components>().insert(
            id, std::forward<Components>(components)
        ), ...
    );
}

void scene::remove(entity id)
{
    for(auto& c: components)
        if(c) c->erase(id);
    if(defer_batch == 0)
        reusable_ids.push_back(id);
    else
        post_batch_reusable_ids.push_back(id);
}

template<typename Component>
void scene::remove(entity id)
{
    get_container<Component>().erase(id);
}

void scene::clear_entities()
{
    for(auto& c: components)
        if(c) c->clear();

    if(defer_batch == 0)
    {
        id_counter = 1;
        reusable_ids.clear();
        post_batch_reusable_ids.clear();
    }
}

void scene::concat(
    scene& other,
    std::map<entity, entity>* translation_table_ptr
){
    std::map<entity, entity> translation_table;

    for(auto& c: other.components)
        if(c) c->list_entities(translation_table);

    start_batch();
    for(auto& pair: translation_table)
        pair.second = add();

    for(auto& c: other.components)
        if(c) c->concat(*this, translation_table);
    finish_batch();

    if(translation_table_ptr)
        *translation_table_ptr = std::move(translation_table);
}

entity scene::copy(scene& other, entity other_id)
{
    entity id = add();

    for(auto& c: other.components)
        if(c) c->copy(*this, id, other_id);

    return id;
}

void scene::start_batch()
{
    ++defer_batch;
    if(defer_batch == 1)
    {
        for(auto& c: components)
            if(c) c->start_batch();
    }
}

void scene::finish_batch()
{
    if(defer_batch > 0)
    {
        --defer_batch;
        if(defer_batch == 0)
        {
            for(auto& c: components)
                if(c) c->finish_batch();

            reusable_ids.insert(
                reusable_ids.end(),
                post_batch_reusable_ids.begin(),
                post_batch_reusable_ids.end()
            );
            post_batch_reusable_ids.clear();
        }
    }
}

template<typename Component>
size_t scene::count() const
{
    return get_container<Component>().size();
}

template<typename Component>
bool scene::has(entity id) const
{
    return get_container<Component>().contains(id);
}

template<typename Component>
const Component* scene::get(entity id) const
{
    return get_container<Component>()[id];
}

template<typename Component>
Component* scene::get(entity id)
{
    return get_container<Component>()[id];
}

template<typename Component, typename... Args>
Component* scene::find_component(Args&&... args)
{
    return get<Component>(
        find<Component>(std::forward<Args>(args)...)
    );
}

template<typename Component, typename... Args>
const Component* scene::find_component(Args&&... args) const
{
    return get<Component>(
        find<Component>(std::forward<Args>(args)...)
    );
}

template<typename Component, typename... Args>
entity scene::find(Args&&... args) const
{
    return get_container<Component>().find_entity(std::forward<Args>(args)...);
}

template<typename Component>
void scene::update_search_index()
{
    return get_container<Component>().update_search_index();
}

void scene::update_search_indices()
{
    for(auto& c: components)
        if(c) c->update_search_index();
}

template<typename EventType>
void scene::emit(const EventType& event)
{
    size_t key = get_event_type_key<EventType>();
    if(event_handlers.size() <= key) return;

    for(event_handler& eh: event_handlers[key])
        eh.callback(*this, &event);
}

template<typename EventType>
size_t scene::get_handler_count() const
{
    size_t key = get_event_type_key<EventType>();
    if(event_handlers.size() <= key) return 0;
    return event_handlers[key].size();
}

template<typename... F>
size_t scene::add_event_handler(F&&... callbacks)
{
    size_t id = subscriber_counter++;
    (internal_add_handler(id, std::forward<F>(callbacks)), ...);
    return id;
}

template<class T, typename... F>
size_t scene::bind_event_handler(T* userdata, F&&... callbacks)
{
    size_t id = subscriber_counter++;
    (internal_bind_handler(id, userdata, std::forward<F>(callbacks)), ...);
    return id;
}

void scene::remove_event_handler(size_t id)
{
    for(std::vector<event_handler>& type_event_handlers: event_handlers)
    {
        for(
            auto it = type_event_handlers.begin();
            it != type_event_handlers.end();
            ++it
        ){
            if(it->subscription_id == id)
            {
                type_event_handlers.erase(it);
                break;
            }
        }
    }
}

template<typename... F>
event_subscription scene::subscribe(F&&... callbacks)
{
    return event_subscription(
        this, add_event_handler(std::forward<F>(callbacks)...)
    );
}

template<typename... EventTypes>
void scene::add_receiver(receiver<EventTypes...>& r)
{
    r.sub.ctx = this;
    r.sub.subscription_id = bind_event_handler(
        &r, &event_receiver<EventTypes>::handle...
    );
}

template<typename Component>
component_container<Component>& scene::get_container() const
{
    size_t key = get_component_type_key<Component>();
    if(components.size() <= key) components.resize(key+1);
    auto& base_ptr = components[key];
    if(!base_ptr)
    {
        base_ptr.reset(new component_container<Component>(*const_cast<scene*>(this)));
        if(defer_batch > 0)
            base_ptr->start_batch();
    }
    return *static_cast<component_container<Component>*>(base_ptr.get());
}

template<typename Component>
size_t scene::get_component_type_key()
{
    static size_t key = component_type_key_counter++;
    return key;
}

template<typename Event>
size_t scene::get_event_type_key()
{
    static size_t key = event_type_key_counter++;
    return key;
}

template<typename F>
void scene::internal_add_handler(size_t id, F&& f)
{
    using T = decltype(event_handler_type_detector(std::function(f)));
    size_t key = get_event_type_key<T>();
    if(event_handlers.size() <= key) event_handlers.resize(key+1);

    event_handler h;
    h.subscription_id = id;
    h.callback = [f = std::forward<F>(f)](scene& ctx, const void* ptr){
        f(ctx, *(const T*)ptr);
    };
    event_handlers[key].push_back(std::move(h));
}

template<class C, typename F>
void scene::internal_bind_handler(size_t id, C* c, F&& f)
{
    using T = decltype(event_handler_type_detector(f));

    size_t key = get_event_type_key<T>();
    if(event_handlers.size() <= key) event_handlers.resize(key+1);

    event_handler h;
    h.subscription_id = id;
    h.callback = [c = c, f = std::forward<F>(f)](scene& ctx, const void* ptr){
        ((*c).*f)(ctx, *(const T*)ptr);
    };
    event_handlers[key].push_back(std::move(h));
}

template<typename... DependencyComponents>
void dependency_components<DependencyComponents...>::
ensure_dependency_components_exist(entity id, scene& ctx)
{
    ((ctx.has<DependencyComponents>(id) ? void() : ctx.attach(id, DependencyComponents())), ...);
}

}
#endif
