import bpy
from io_scene_gltf2.blender.exp import gltf2_blender_gather_image

bl_info = {
    "name": "Tauray glTF extension",
    "category": "Generic",
    "version": (1, 0, 1),
    "blender": (3, 0, 0),
    'location': 'File > Export > glTF 2.0',
    'description': 'Add-on to add Tauray data to an exported glTF file.',
    'isDraft': False,
    'developer': "Julius Ikkala",
    'url': 'julius.ikkala@tuni.fi',
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
            if probe.type == 'GRID':
                probe_data["resolution_x"] = probe.grid_resolution_x
                probe_data["resolution_y"] = probe.grid_resolution_z
                probe_data["resolution_z"] = probe.grid_resolution_y
            probe_data["radius"] = probe.influence_distance
            probe_data["clip_near"] = probe.clip_start
            probe_data["clip_far"] = probe.clip_end
            data["light_probe"] = probe_data

        if blender_object.type == 'MESH':
            mesh_data = {}
            mesh_data["shadow_terminator_offset"] = blender_object.cycles.shadow_terminator_offset
            data["mesh"] = mesh_data

        gltf2_object.extensions[glTF_extension_name] = self.Extension(
            name=glTF_extension_name,
            extension=data,
            required=extension_is_required
        )

    def gather_material_pbr_metallic_roughness_hook(self, gltf2_material, blender_material, orm_texture, export_settings):
        if not self.properties.enabled:
            return

        if gltf2_material.extensions is None:
            gltf2_material.extensions = {}

        principled_bsdf = None
        for node in blender_material.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled_bsdf = node
                break

        if principled_bsdf is None:
            return

        transmission_input = None
        emission_input = None
        emission_strength = None
        for inp in principled_bsdf.inputs:
            if inp.identifier == 'Transmission':
                transmission_input = inp.default_value
            if inp.identifier == 'Emission':
                emission_input = list(inp.default_value)
            if inp.identifier == 'Emission Strength':
                emission_strength = inp.default_value

        data = {}

        if transmission_input is not None:
            data["transmission"] = transmission_input

        if emission_input is not None:
            if emission_strength is not None:
                emission_input = [n * emission_strength for n in emission_input]
            data["emission"] = emission_input

        gltf2_material.extensions[glTF_extension_name] = self.Extension(
            name=glTF_extension_name,
            extension=data,
            required=extension_is_required
        )

    def gather_scene_hook(self, gltf2_scene, blender_scene, export_settings):
        if not self.properties.enabled:
            return

        if gltf2_scene.extensions is None:
            gltf2_scene.extensions = {}

        nodes = blender_scene.world.node_tree.nodes
        links = blender_scene.world.node_tree.links
        background_node = None
        envmap_node = None
        for node in nodes:
            if node.type == 'TEX_ENVIRONMENT':
                envmap_node = node
            elif node.type == 'BACKGROUND':
                background_node = node

        if background_node is None or envmap_node is None:
            return

        # A bit of a hack that hopefully won't hurt anyone ;)
        img_node = nodes.new('ShaderNodeTexImage')
        img_node.image = envmap_node.image
        img_node.interpolation = envmap_node.interpolation
        img_node.projection = 'FLAT'
        tmp_link = links.new(background_node.inputs[0], img_node.outputs[0])

        img = gltf2_blender_gather_image.gather_image(
            (background_node.inputs[0],), export_settings)
        data = {
            'envmap': img,
            'envmap_factor': background_node.inputs[1].default_value
        }
        links.remove(tmp_link)
        nodes.remove(img_node)

        links.new(background_node.inputs[0], envmap_node.outputs[0])

        gltf2_scene.extensions[glTF_extension_name] = self.Extension(
            name=glTF_extension_name,
            extension=data,
            required=extension_is_required
        )
