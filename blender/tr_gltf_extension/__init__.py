import bpy

bl_info = {
    "name": "Tauray glTF extension",
    "category": "Generic",
    "version": (1, 2, 1),
    "blender": (5, 2, 2),
    'location': 'File > Export > glTF 2.0',
    'description': 'Add-on to add Tauray data to an exported glTF file.',
    'isDraft': False,
    'developer': "Julius Ikkala / Tampere University",
    'url': 'julius.ikkala@gmail.com',
}

glTF_extension_name = "TR_data"

extension_is_required = False

class TRExtensionProperties(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(
        name=bl_info["name"],
        description='Include Tauray data in the exported glTF file.',
        default=True
        )

def register():
    bpy.utils.register_class(TRExtensionProperties)
    bpy.types.Scene.TRExtensionProperties = bpy.props.PointerProperty(type=TRExtensionProperties)

def register_panel():
    try:
        bpy.utils.register_class(GLTF_PT_UserExtensionPanel)
    except Exception:
        pass

    return unregister_panel


def unregister_panel():
    try:
        bpy.utils.unregister_class(GLTF_PT_UserExtensionPanel)
    except Exception:
        pass


def unregister():
    unregister_panel()
    bpy.utils.unregister_class(TRExtensionProperties)
    del bpy.types.Scene.TRExtensionProperties

class GLTF_PT_UserExtensionPanel(bpy.types.Panel):

    bl_space_type = 'FILE_BROWSER'
    bl_region_type = 'TOOL_PROPS'
    bl_label = "Enabled"
    bl_parent_id = "GLTF_PT_export_user_extensions"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        sfile = context.space_data
        operator = sfile.active_operator
        return operator.bl_idname == "EXPORT_SCENE_OT_gltf"

    def draw_header(self, context):
        props = bpy.context.scene.TRExtensionProperties
        self.layout.prop(props, 'enabled')

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        props = bpy.context.scene.TRExtensionProperties
        layout.active = props.enabled

        box = layout.box()
        box.label(text=glTF_extension_name)

        props = bpy.context.scene.TRExtensionProperties


class glTF2ExportUserExtension:

    def __init__(self):
        from io_scene_gltf2.io.com.gltf2_io_extensions import Extension
        self.Extension = Extension
        self.properties = bpy.context.scene.TRExtensionProperties

    def gather_node_hook(self, gltf2_object, blender_object, export_settings):
        if not self.properties.enabled:
            return
        if gltf2_object.extensions is None:
            gltf2_object.extensions = {}

        data = {}

        if blender_object.type == 'LIGHT':
            light = blender_object.data
            light_data = {}
            if light.type == 'POINT' or light.type == 'SPOT':
                light_data["radius"] = light.shadow_soft_size
            elif light.type == 'SUN':
                # In radians, max angle from direction that is still lit.
                light_data["angle"] = light.angle/2
            data["light"] = light_data

        if blender_object.type == 'LIGHT_PROBE':
            probe = blender_object.data
            probe_data = {}
            probe_data["type"] = probe.type
            if probe.type == 'VOLUME':
                probe_data["resolution_x"] = probe.resolution_x
                probe_data["resolution_y"] = probe.resolution_z
                probe_data["resolution_z"] = probe.resolution_y
            probe_data["radius"] = probe.influence_distance
            data["light_probe"] = probe_data

        gltf2_object.extensions[glTF_extension_name] = self.Extension(
            name=glTF_extension_name,
            extension=data,
            required=extension_is_required
        )
