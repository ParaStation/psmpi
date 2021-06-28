HWLOC_DECLSPEC extern const struct hwloc_component hwloc_noos_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_linux_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_synthetic_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_xml_component;
HWLOC_DECLSPEC extern const struct hwloc_component hwloc_xml_nolibxml_component;

static const struct hwloc_component * hwloc_static_components[] = {
  &hwloc_noos_component,
  &hwloc_linux_component,
  &hwloc_synthetic_component,
  &hwloc_xml_component,
  &hwloc_xml_nolibxml_component,
  NULL
};
