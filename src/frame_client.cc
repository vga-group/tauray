#include "frame_client.hh"
#include <nng/nng.h>
#include <nng/protocol/bus0/bus.h>

namespace tr
{

void upload_texture(nng_msg* msg, SDL_Texture* tex, int width, int height)
{
    uint8_t* src = (uint8_t*)nng_msg_body(msg);
    uint8_t* dst = nullptr;
    int pitch = 0;
    SDL_LockTexture(tex, nullptr, reinterpret_cast<void **>(&dst), &pitch);

    memcpy(dst, src, width*height*3);

    SDL_UnlockTexture(tex);
    nng_msg_free(msg);
}

void frame_client(const options& opt)
{
    SDL_Init(SDL_INIT_EVENTS|SDL_INIT_VIDEO);

    uint32_t width = opt.width;
    uint32_t height = opt.height;
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
    SDL_Window* win = SDL_CreateWindow(
        "Tauray", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        opt.width, opt.height,
        (opt.fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0)
    );
    SDL_Renderer* ren = SDL_CreateRenderer(
        win, -1, SDL_RENDERER_ACCELERATED
    );
    SDL_SetRelativeMouseMode((SDL_bool)true);
    SDL_SetWindowGrab(win, (SDL_bool)true);
    SDL_ShowCursor(SDL_DISABLE);

    SDL_Texture* tex = SDL_CreateTexture(
        ren, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, opt.width, opt.height
    );

    nng_socket socket;
    nng_bus0_open(&socket);
    std::string address = "tcp://"+opt.connect;
    nng_dial(socket, address.c_str(), nullptr, NNG_FLAG_NONBLOCK);

    using namespace std::chrono_literals;
    auto last_request_timestamp = std::chrono::steady_clock::now();

    bool running = true;
    std::vector<SDL_Event> new_events;
    while(running)
    {
        SDL_Event event;
        while(SDL_PollEvent(&event)) switch(event.type)
        {
        case SDL_QUIT:
            running = false;
            break;
        case SDL_KEYDOWN:
            if(event.key.keysym.sym == SDLK_ESCAPE)
            {
                running = false;
                break;
            }
            [[fallthrough]];
        default:
            new_events.push_back(event);
            break;
        }

        if(!running)
            break;

        auto timestamp = std::chrono::steady_clock::now();
        auto duration_since_last_request = timestamp - last_request_timestamp;
        bool timeout_close = duration_since_last_request > 0.5s;

        if(new_events.size() != 0 || timeout_close)
        {
            last_request_timestamp = timestamp;
            nng_msg* msg = nullptr;
            nng_msg_alloc(&msg, 0);
            nng_msg_append(msg, new_events.data(), sizeof(SDL_Event)*new_events.size());
            if(nng_sendmsg(socket, msg, 0) != 0)
                nng_msg_free(msg);
            new_events.clear();
        }

        nng_msg* msg = nullptr;
        int result = nng_recvmsg(socket, &msg, NNG_FLAG_NONBLOCK);
        if(result == 0)
        {
            uint32_t channels = 0;
            uint32_t new_width = 0;
            uint32_t new_height = 0;
            nng_msg_trim_u32(msg, &new_width);
            nng_msg_trim_u32(msg, &new_height);
            nng_msg_trim_u32(msg, &channels);

            if(new_width <= 16384 && new_height <= 16384)
            {
                if(new_width != width || new_height != height)
                {
                    width = new_width;
                    height = new_height;
                    SDL_DestroyTexture(tex);
                    tex = SDL_CreateTexture(
                        ren, SDL_PIXELFORMAT_RGB24,
                        SDL_TEXTUREACCESS_STREAMING, width, height
                    );
                }

                upload_texture(msg, tex, width, height);

                SDL_RenderCopy(ren, tex, nullptr, nullptr);

                SDL_RenderPresent(ren);
            }
            else nng_msg_free(msg);
        }
    }

    nng_close(socket);

    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
}

}
