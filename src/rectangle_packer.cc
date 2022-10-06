#include "rectangle_packer.hh"
#include <algorithm>
#include <cmath>

namespace
{
    int calc_overlap(int x1, int w1, int x2, int w2)
    {
        return std::max(std::min(x1 + w1, x2 + w2) - std::max(x1, x2), 0);
    }

    int get_cell_size(int total_area)
    {
        // Largely empirical, I tested different cell sizes for different sizes
        // of squares. This equation mostly follows the resulting values.
        return ceil(pow(total_area, 1.0/6.0));
    }
}

namespace tr
{

rect_packer::rect_packer(int w, int h, bool open)
: canvas_w(w), canvas_h(h), cell_size(16), open(open), marker(0)
{
    reset(w, h);
}

void rect_packer::enlarge(int w, int h)
{
    tmp.clear();

    std::vector<free_edge> top_edges, right_edges;

    w = std::max(canvas_w, w);
    h = std::max(canvas_h, h);

    marker++;
    if(h > canvas_h)
    {
        top_edges.push_back({0, canvas_h, canvas_w, false, true, marker});

        for(int i = 0; i < lookup_w; ++i)
        {
            auto& cell = edge_lookup[(lookup_h-1) * lookup_w + i];
            for(free_edge* edge: cell)
            {
                if(
                    edge->vertical ||
                    edge->y != canvas_h ||
                    edge->marker == marker
                ) continue;
                edge->marker = marker;
                edge_clip(*edge, top_edges);
                tmp.push_back(edge);
            }
        }

        top_edges.push_back({0, canvas_h, h-canvas_h, true, true, marker});
        top_edges.push_back({0, h, w, false, false, marker});
        if(w <= canvas_w)
            top_edges.push_back({w, canvas_h, h-canvas_h, true, false, marker});
    }

    if(w > canvas_w)
    {
        right_edges.push_back({canvas_w, 0, canvas_h, true, true, marker});

        for(int i = 0; i < lookup_h; ++i)
        {
            auto& cell = edge_lookup[i * lookup_w + lookup_w - 1];
            for(free_edge* edge: cell)
            {
                if(
                    !edge->vertical ||
                    edge->x != canvas_w ||
                    edge->marker == marker
                ) continue;
                edge->marker = marker;
                edge_clip(*edge, right_edges);
                tmp.push_back(edge);
            }
        }

        right_edges.push_back({canvas_w, 0, w-canvas_w, false, true, marker});
        right_edges.push_back({w, 0, h, true, false, marker});
        if(h <= canvas_h)
            right_edges.push_back(
                {canvas_w, h, w-canvas_w, false, false, marker}
            );
    }

    std::sort(tmp.begin(), tmp.end());
    free_edge* base = edges.data();
    for(free_edge* edge: tmp)
    {
        edges.erase(edges.begin()+(edge-base));
        base++;
    }

    edges.insert(edges.end(), top_edges.begin(), top_edges.end());
    edges.insert(edges.end(), right_edges.begin(), right_edges.end());

    canvas_h = h;
    canvas_w = w;

    set_cell_size();
}

void rect_packer::reset(int w, int h)
{
    canvas_w = w;
    canvas_h = h;
    lookup_w = (canvas_w+cell_size-1)/cell_size;
    lookup_h = (canvas_h+cell_size-1)/cell_size;
    reset();
}

void rect_packer::reset()
{
    edge_lookup.resize(lookup_w*lookup_h);
    for(auto& cell: edge_lookup)
        cell.clear();

    edges.clear();
    edges.push_back({0, 0, canvas_h, true, true, marker});
    edges.push_back({0, 0, canvas_w, false, true, marker});
    edges.push_back({canvas_w, 0, canvas_h, true, false, marker});
    edges.push_back({0, canvas_h, canvas_w, false, false, marker});
    recalc_edge_lookup();
}

void rect_packer::set_cell_size(int cell_size)
{
    if(cell_size < 1) cell_size = get_cell_size(canvas_w*canvas_h);
    this->cell_size = cell_size;

    lookup_h = (canvas_h+cell_size-1)/cell_size;
    lookup_w = (canvas_w+cell_size-1)/cell_size;

    edge_lookup.resize(lookup_w*lookup_h);
    for(auto& cell: edge_lookup)
        cell.clear();

    recalc_edge_lookup();
}

void rect_packer::set_open(bool open)
{
    this->open = open;
}

bool rect_packer::pack(int w, int h, int& x, int& y)
{
    std::vector<free_edge*> affected;

    int score = 0;
    score = find_max_score(w, h, x, y, affected);

    // No fit, fail.
    if(score == 0) return false;

    place_rect(x, y, w, h, affected);

    return true;
}

bool rect_packer::pack_rotate(int w, int h, int& x, int& y, bool& rotated)
{
    // Fast path if we rotation is meaningless.
    if(w == h)
    {
        rotated = false;
        return pack(w, h, x, y);
    }

    // Try both orientations.
    int rot_x, rot_y;
    std::vector<free_edge*> affected, rot_affected;
    int score = 0;
    int rot_score = 0;

    score = find_max_score(w, h, x, y, affected);
    rot_score = find_max_score(h, w, rot_x, rot_y, rot_affected);
    if(score == 0 && rot_score == 0) return false;

    // Pick better orientation, preferring non-rotated version.
    if(score >= rot_score)
    {
        rotated = false;
        place_rect(x, y, w, h, affected);
    }
    else
    {
        rotated = true;
        x = rot_x; y = rot_y;
        place_rect(x, y, h, w, rot_affected);
    }

    return true;
}

int rect_packer::pack(rect* rects, size_t count, bool allow_rotation)
{
    int packed = 0;

    std::vector<rect*> rr;
    rr.resize(count);
    for(unsigned i = 0; i < count; ++i)
    {
        rr[i] = rects + i;
        rr[i]->rotated = false;
    }

    std::sort(
        rr.begin(),
        rr.end(),
        [](const rect* a, const rect* b){
            return std::max(a->w, a->h) > std::max(b->w, b->h);
        }
    );

    for(rect* r: rr)
    {
        if(r->packed)
        {
            packed++;
            continue;
        }

        if(allow_rotation)
        {
            if(pack_rotate(r->w, r->h, r->x, r->y, r->rotated))
            {
                r->packed = true;
                packed++;
            }
        }
        else
        {
            if(pack(r->w, r->h, r->x, r->y))
            {
                r->packed = true;
                packed++;
            }
        }
    }
    return packed;
}

void rect_packer::recalc_edge_lookup()
{
    marker = 0;

    // Clear lookup
    for(auto& cell: edge_lookup)
        cell.clear();

    // Rasterize edges on the lookup
    for(auto& edge: edges)
    {
        edge.marker = 0;

        int sx = edge.x/cell_size;
        int bx = edge.x%cell_size;
        int sy = edge.y/cell_size;
        int by = edge.y%cell_size;

        if(edge.vertical)
        {
            int ey = (edge.y+edge.length-1)/cell_size;
            bool border = bx == 0 && sx > 0;

            for(; sy <= ey; ++sy)
            {
                if(sx < lookup_w)
                    edge_lookup[sy * lookup_w + sx].push_back(&edge);
                if(border)
                    edge_lookup[sy * lookup_w + sx-1].push_back(&edge);
            }
        }
        else
        {
            int ex = (edge.x+edge.length-1)/cell_size;
            bool border = by == 0 && sy > 0;

            for(; sx <= ex; ++sx)
            {
                if(sy < lookup_h)
                    edge_lookup[sy * lookup_w + sx].push_back(&edge);
                if(border)
                    edge_lookup[(sy-1) * lookup_w + sx].push_back(&edge);
            }
        }
    }
}

int rect_packer::find_max_score(
    int w, int h, int& best_x, int& best_y,
    std::vector<free_edge*>& best_affected_edges
){
    int best_score = 0;
    int ideal = (w + h) * 2;
    for(free_edge& edge: edges)
    {
        if(edge.vertical)
        {
            int x = edge.x;
            if(!edge.up_right_inside) x -= w;
            if(x < 0 || x + w > canvas_w) continue;

            int ey = std::min(edge.y + edge.length, canvas_h - h + 1);

            for(int y = edge.y; y < ey;)
            {
                int skip = edge.vertical;
                int score = score_rect(x, y, w, h, skip, ey, tmp);
                if(score > best_score)
                {
                    best_score = score;
                    best_x = x;
                    best_y = y;
                    best_affected_edges = tmp;
                }
                y += skip;
            }
        }
        else
        {
            int y = edge.y;
            if(!edge.up_right_inside) y -= h;
            if(y < 0 || y + h > canvas_h) continue;

            int ex = std::min(edge.x + edge.length, canvas_w - w + 1);

            for(int x = edge.x; x < ex;)
            {
                int skip = edge.vertical;
                int score = score_rect(x, y, w, h, skip, ex, tmp);
                if(score > best_score)
                {
                    best_score = score;
                    best_x = x;
                    best_y = y;
                    best_affected_edges = tmp;
                }
                x += skip;
            }
        }
        if(best_score == ideal) break;
    }
    return best_score;
}

int rect_packer::score_rect(
    int x, int y, int w, int h, int& skip, int end,
    std::vector<free_edge*>& affected_edges
){
    affected_edges.clear();

    bool vertical = skip;
    int score = 0;
    int sx = x/cell_size;
    int sy = y/cell_size;
    int ex = (x+w-1)/cell_size;
    int ey = (y+h-1)/cell_size;

    if(vertical) end = std::min(end, (ey+1)*cell_size);
    else end = std::min(end, (ex+1)*cell_size);

    marker++;
    for(int cy = sy; cy <= ey; ++cy)
    {
        for(int cx = sx; cx <= ex; ++cx)
        {
            auto& cell = edge_lookup[cy * lookup_w + cx];
            for(free_edge* edge: cell)
            {
                if(edge->marker == marker) continue;
                edge->marker = marker;

                int escore = score_rect_edge(x, y, w, h, edge); 
                if(escore == -1)
                {
                    if(vertical) skip = edge->y + edge->length - y;
                    else skip = edge->x + edge->length - x;
                    return 0;
                }

                if(escore > 0)
                {
                    affected_edges.push_back(edge);
                    score += escore;
                }

                if(vertical)
                {
                    if(edge->vertical && edge->x == x + w && edge->y > y)
                        end = std::min(end, edge->y);
                    else if(
                        !edge->vertical && edge->y > y + h &&
                        edge->x < x + w && edge->x + edge->length > x
                    ) end = std::min(end, edge->y - h);
                }
                else
                {
                    if(!edge->vertical && edge->y == y + h && edge->x > x)
                        end = std::min(end, edge->x);
                    else if(
                        edge->vertical && edge->x > x + w &&
                        edge->y < y + h && edge->y + edge->length > y
                    ) end = std::min(end, edge->x - w);
                }
            }
        }
    }
    //skip = 1;
    if(vertical) skip = end - y;
    else skip = end - x;
    return score;
}

int rect_packer::score_rect_edge(
    int x, int y, int w, int h, free_edge* edge
){
    if(edge->vertical)
    {
        int score = calc_overlap(y, h, edge->y, edge->length);
        if(edge->x > x && edge->x < x + w && score > 0) return -1;
        if(open && edge->x == canvas_w) return 0;
        if(x == edge->x || x + w == edge->x) return score;
    }
    else
    {
        int score = calc_overlap(x, w, edge->x, edge->length);
        if(edge->y > y && edge->y < y + h && score > 0) return -1;
        if(open && edge->y == canvas_h) return 0;
        if(y == edge->y || y + h == edge->y) return score;
    }
    return -2;
}

// This function doesn't have to be super optimized in terms of allocations,
// it's run only once when packing a rect.
void rect_packer::place_rect(
    int x, int y, int w, int h,
    std::vector<free_edge*>& affected_edges
){
    std::vector<free_edge> new_edges;
    std::vector<free_edge*> delete_edges;
    std::vector<free_edge> vert_rect_edges;
    std::vector<free_edge> hori_rect_edges;

    vert_rect_edges.push_back({x,y,h,true,false,marker});
    vert_rect_edges.push_back({x+w,y,h,true,true,marker});

    hori_rect_edges.push_back({x,y,w,false,false,marker});
    hori_rect_edges.push_back({x,y+h,w,false,true,marker});

    for(free_edge* edge: affected_edges)
    {
        free_edge a, b;

        if(edge->vertical)
        {
            a = {
                edge->x, edge->y, y - edge->y,
                true, edge->up_right_inside, marker
            };
            b = {
                edge->x, y + h, edge->y + edge->length - y - h,
                true, edge->up_right_inside, marker
            };
            edge_clip(*edge, vert_rect_edges);
        }
        else
        {
            a = {
                edge->x, edge->y, x - edge->x,
                false, edge->up_right_inside, marker
            };
            b = {
                x + w, edge->y, edge->x + edge->length - x - w,
                false, edge->up_right_inside, marker
            };
            edge_clip(*edge, hori_rect_edges);
        }

        if(a.length > 0 && b.length > 0)
        {
            *edge = a;
            new_edges.push_back(b);
        }
        else if(a.length > 0) *edge = a;
        else if(b.length > 0) *edge = b;
        else delete_edges.push_back(edge);
    }

    std::sort(delete_edges.begin(), delete_edges.end());
    free_edge* base = edges.data();
    for(free_edge* edge: delete_edges)
    {
        edges.erase(edges.begin()+(edge-base));
        base++;
    }

    edges.insert(edges.end(), new_edges.begin(), new_edges.end());
    edges.insert(edges.end(), vert_rect_edges.begin(), vert_rect_edges.end());
    edges.insert(edges.end(), hori_rect_edges.begin(), hori_rect_edges.end());

    recalc_edge_lookup();
}

void rect_packer::edge_clip(
    const free_edge& mask,
    std::vector<free_edge>& clipped
){
    for(unsigned i = 0; i < clipped.size(); ++i)
    {
        free_edge* edge = &clipped[i];

        free_edge a, b;
        if(mask.vertical)
        {
            if(mask.x != edge->x) continue;
            a = {
                edge->x, edge->y, std::min(mask.y - edge->y, edge->length),
                true, edge->up_right_inside, marker
            };
            b = {
                edge->x, std::max(mask.y + mask.length, edge->y),
                edge->y + edge->length,
                true, edge->up_right_inside, marker
            };
            b.length -= b.y;
        }
        else
        {
            if(mask.y != edge->y) continue;
            a = {
                edge->x, edge->y, std::min(mask.x - edge->x, edge->length),
                false, edge->up_right_inside, marker
            };
            b = {
                std::max(mask.x + mask.length, edge->x), edge->y,
                edge->x + edge->length,
                false, edge->up_right_inside, marker
            };
            b.length -= b.x;
        }

        if(a.length > 0 && b.length > 0)
        {
            *edge = a;
            clipped.push_back(b);
        }
        else if(a.length > 0) *edge = a;
        else if(b.length > 0) *edge = b;
        else {
            clipped.erase(clipped.begin()+i);
            --i;
        }
    }
}

}
