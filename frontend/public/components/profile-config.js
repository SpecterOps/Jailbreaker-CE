export const CONFIGURE_NEW_VALUE = "__configure_new__";

const SECRET_CONFIG_KEYS = ["api_key", "attacker_api_key", "judge_api_key"];

export function fillProfileSelect(select, profiles, purpose, savedValue = "", options = {}) {
  if (!select) return;

  const {
    blankLabel = "Manual / session key",
    includeBlank = true,
    configureLabel = "",
    beforeOptions = [],
    fallbackValue = includeBlank ? "" : beforeOptions[0]?.value || "",
  } = options;

  const preferredValues = [
    ...(select.value ? [select.value] : []),
    ...(savedValue ? [savedValue] : []),
    fallbackValue,
  ];
  const optionNodes = [
    ...beforeOptions.map((option) => createSelectOption(option.value, option.label)),
    ...(includeBlank ? [createSelectOption("", blankLabel)] : []),
    ...(profiles || [])
      .map(createProfileOption),
  ];

  if (configureLabel) {
    optionNodes.push(createSelectOption(CONFIGURE_NEW_VALUE, configureLabel));
  }

  select.replaceChildren(...optionNodes);
  const values = new Set(optionNodes.map((option) => option.value));
  select.value = preferredValues.find((value) => values.has(value)) ?? fallbackValue;
}

export function selectedProfileFromList(profiles, select) {
  const profileId = select?.value || "";
  return (profiles || []).find((profile) => profile.profile_id === profileId) || null;
}

export function stripSecretConfigKeys(config) {
  config = { ...(config || {}) };
  delete config.api_key;
  delete config.attacker_api_key;
  delete config.judge_api_key;
  return config;
}

export function hasSecretConfigKeys(config) {
  return SECRET_CONFIG_KEYS.some((key) => Object.prototype.hasOwnProperty.call(config || {}, key));
}

function createSelectOption(value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  return option;
}

function createProfileOption(profile) {
  const option = createSelectOption(profile.profile_id, `${profile.name} · ${profile.model}`);
  option.dataset.baseUrl = profile.base_url || "";
  option.dataset.model = profile.model || "";
  return option;
}
