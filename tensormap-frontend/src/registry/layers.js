const ACTIVATION_OPTIONS = [
  { value: "none", label: "None" },
  { value: "relu", label: "ReLU" },
  { value: "sigmoid", label: "Sigmoid" },
  { value: "tanh", label: "Tanh" },
  { value: "softmax", label: "Softmax" },
  { value: "elu", label: "ELU" },
  { value: "selu", label: "SELU" },
];

const PADDING_OPTIONS = [
  { value: "valid", label: "Valid" },
  { value: "same", label: "Same" },
];

const INITIALIZER_OPTIONS = [
  { value: "glorot_uniform", label: "Glorot Uniform" },
  { value: "he_normal", label: "He Normal" },
  { value: "lecun_normal", label: "Lecun Normal" },
];

export const LAYER_REGISTRY = [
  {
    type: "custominput",
    label: "Input",
    category: "Core",
    accentClass: "border-l-node-input",
    headerClass: "bg-node-input",
    params: {
      "dim-1": {
        label: "Dim 1",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Dimension 1",
      },
      "dim-2": {
        label: "Dim 2",
        type: "number",
        required: false,
        min: 1,
        placeholder: "Dimension 2 (optional)",
      },
      "dim-3": {
        label: "Dim 3",
        type: "number",
        required: false,
        min: 1,
        placeholder: "Dimension 3 (optional)",
      },
    },
  },
  {
    type: "customdense",
    label: "Dense",
    category: "Core",
    accentClass: "border-l-node-dense",
    headerClass: "bg-node-dense",
    params: {
      units: {
        label: "Units",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Number of units",
      },
      activation: {
        label: "Activation",
        type: "select",
        required: true,
        options: ACTIVATION_OPTIONS,
        defaultValue: "relu",
      },
    },
  },
  {
    type: "customflatten",
    label: "Flatten",
    category: "Core",
    accentClass: "border-l-node-flatten",
    headerClass: "bg-node-flatten",
    params: {},
  },
  {
    type: "customconv",
    label: "Conv2D",
    category: "CNN",
    accentClass: "border-l-node-cnn",
    headerClass: "bg-node-cnn",
    params: {
      filter: {
        label: "Filters",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Filter count",
      },
      kernelX: {
        label: "Kernel X",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Kernel X",
      },
      kernelY: {
        label: "Kernel Y",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Kernel Y",
      },
      strideX: {
        label: "Stride X",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Stride X",
      },
      strideY: {
        label: "Stride Y",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Stride Y",
      },
      padding: {
        label: "Padding",
        type: "select",
        required: false,
        options: PADDING_OPTIONS,
        defaultValue: "valid",
      },
      activation: {
        label: "Activation",
        type: "select",
        required: false,
        options: ACTIVATION_OPTIONS,
        defaultValue: "none",
      },
    },
  },
  {
    type: "custommaxpool2d",
    label: "MaxPooling2D",
    category: "CNN",
    accentClass: "border-l-node-cnn",
    headerClass: "bg-node-cnn",
    params: {
      poolX: {
        label: "Pool X",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Pool size X",
      },
      poolY: {
        label: "Pool Y",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Pool size Y",
      },
      strideX: {
        label: "Stride X",
        type: "number",
        required: false,
        min: 1,
        placeholder: "Stride X (optional)",
      },
      strideY: {
        label: "Stride Y",
        type: "number",
        required: false,
        min: 1,
        placeholder: "Stride Y (optional)",
      },
      padding: {
        label: "Padding",
        type: "select",
        required: false,
        options: PADDING_OPTIONS,
        defaultValue: "valid",
      },
    },
  },
  {
    type: "customavgpool2d",
    label: "AveragePooling2D",
    category: "CNN",
    accentClass: "border-l-node-cnn",
    headerClass: "bg-node-cnn",
    params: {
      poolX: {
        label: "Pool X",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Pool size X",
      },
      poolY: {
        label: "Pool Y",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Pool size Y",
      },
      strideX: {
        label: "Stride X",
        type: "number",
        required: false,
        min: 1,
        placeholder: "Stride X (optional)",
      },
      strideY: {
        label: "Stride Y",
        type: "number",
        required: false,
        min: 1,
        placeholder: "Stride Y (optional)",
      },
      padding: {
        label: "Padding",
        type: "select",
        required: false,
        options: PADDING_OPTIONS,
        defaultValue: "valid",
      },
    },
  },
  {
    type: "customglobalavgpool2d",
    label: "GlobalAveragePooling2D",
    category: "CNN",
    accentClass: "border-l-node-cnn",
    headerClass: "bg-node-cnn",
    params: {},
  },
  {
    type: "customsepconv2d",
    label: "SeparableConv2D",
    category: "CNN",
    accentClass: "border-l-node-cnn",
    headerClass: "bg-node-cnn",
    params: {
      filter: {
        label: "Filters",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Filter count",
      },
      kernelX: {
        label: "Kernel X",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Kernel X",
      },
      kernelY: {
        label: "Kernel Y",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Kernel Y",
      },
      strideX: {
        label: "Stride X",
        type: "number",
        required: false,
        min: 1,
        placeholder: "Stride X (optional)",
      },
      strideY: {
        label: "Stride Y",
        type: "number",
        required: false,
        min: 1,
        placeholder: "Stride Y (optional)",
      },
      depthMultiplier: {
        label: "Depth Multiplier",
        type: "number",
        required: false,
        min: 1,
        placeholder: "Depth multiplier (optional)",
      },
      padding: {
        label: "Padding",
        type: "select",
        required: false,
        options: PADDING_OPTIONS,
        defaultValue: "valid",
      },
      activation: {
        label: "Activation",
        type: "select",
        required: false,
        options: ACTIVATION_OPTIONS,
        defaultValue: "none",
      },
    },
  },
  {
    type: "customembedding",
    label: "Embedding",
    category: "RNN",
    accentClass: "border-l-node-rnn",
    headerClass: "bg-node-rnn",
    params: {
      input_dim: {
        label: "Input Dim",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Vocabulary size",
      },
      output_dim: {
        label: "Output Dim",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Embedding size",
      },
      input_length: {
        label: "Input Length",
        type: "number",
        required: false,
        min: 1,
        placeholder: "Sequence length (optional)",
      },
    },
  },
  {
    type: "customlstm",
    label: "LSTM",
    category: "RNN",
    accentClass: "border-l-node-rnn",
    headerClass: "bg-node-rnn",
    params: {
      units: {
        label: "Units",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Hidden units",
      },
      activation: {
        label: "Activation",
        type: "select",
        required: false,
        options: ACTIVATION_OPTIONS,
        defaultValue: "tanh",
      },
      recurrent_activation: {
        label: "Recurrent Activation",
        type: "select",
        required: false,
        options: ACTIVATION_OPTIONS.filter((opt) => opt.value !== "softmax"),
        defaultValue: "sigmoid",
      },
      return_sequences: {
        label: "Return Sequences",
        type: "boolean",
        required: false,
        defaultValue: false,
      },
      dropout: {
        label: "Dropout",
        type: "number",
        required: false,
        min: 0,
        max: 1,
        step: 0.05,
        placeholder: "0 - 1",
      },
      recurrent_dropout: {
        label: "Recurrent Dropout",
        type: "number",
        required: false,
        min: 0,
        max: 1,
        step: 0.05,
        placeholder: "0 - 1",
      },
    },
  },
  {
    type: "customgru",
    label: "GRU",
    category: "RNN",
    accentClass: "border-l-node-rnn",
    headerClass: "bg-node-rnn",
    params: {
      units: {
        label: "Units",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Hidden units",
      },
      activation: {
        label: "Activation",
        type: "select",
        required: false,
        options: ACTIVATION_OPTIONS,
        defaultValue: "tanh",
      },
      recurrent_activation: {
        label: "Recurrent Activation",
        type: "select",
        required: false,
        options: ACTIVATION_OPTIONS.filter((opt) => opt.value !== "softmax"),
        defaultValue: "sigmoid",
      },
      return_sequences: {
        label: "Return Sequences",
        type: "boolean",
        required: false,
        defaultValue: false,
      },
      dropout: {
        label: "Dropout",
        type: "number",
        required: false,
        min: 0,
        max: 1,
        step: 0.05,
        placeholder: "0 - 1",
      },
      recurrent_dropout: {
        label: "Recurrent Dropout",
        type: "number",
        required: false,
        min: 0,
        max: 1,
        step: 0.05,
        placeholder: "0 - 1",
      },
    },
  },
  {
    type: "customdropout",
    label: "Dropout",
    category: "Regularization",
    accentClass: "border-l-node-regularization",
    headerClass: "bg-node-regularization",
    params: {
      rate: {
        label: "Rate",
        type: "number",
        required: true,
        min: 0,
        max: 1,
        step: 0.05,
        placeholder: "0 - 1",
      },
    },
  },
  {
    type: "custombatchnorm",
    label: "BatchNormalization",
    category: "Regularization",
    accentClass: "border-l-node-regularization",
    headerClass: "bg-node-regularization",
    params: {
      momentum: {
        label: "Momentum",
        type: "number",
        required: false,
        min: 0,
        max: 1,
        step: 0.01,
        placeholder: "0 - 1",
        defaultValue: 0.99,
      },
      epsilon: {
        label: "Epsilon",
        type: "number",
        required: false,
        min: 0,
        step: 0.0001,
        placeholder: "Small constant",
        defaultValue: 0.001,
      },
    },
  },
  {
    type: "customactivation",
    label: "Activation",
    category: "Dense Variants",
    accentClass: "border-l-node-activation",
    headerClass: "bg-node-activation",
    params: {
      activation: {
        label: "Activation",
        type: "select",
        required: true,
        options: ACTIVATION_OPTIONS,
        defaultValue: "relu",
      },
    },
  },
  {
    type: "customdenseadvanced",
    label: "Dense (Advanced)",
    category: "Dense Variants",
    accentClass: "border-l-node-activation",
    headerClass: "bg-node-activation",
    params: {
      units: {
        label: "Units",
        type: "number",
        required: true,
        min: 1,
        placeholder: "Number of units",
      },
      activation: {
        label: "Activation",
        type: "select",
        required: false,
        options: ACTIVATION_OPTIONS,
        defaultValue: "relu",
      },
      use_bias: {
        label: "Use Bias",
        type: "boolean",
        required: false,
        defaultValue: true,
      },
      kernel_initializer: {
        label: "Kernel Initializer",
        type: "select",
        required: false,
        options: INITIALIZER_OPTIONS,
        defaultValue: "glorot_uniform",
      },
    },
  },
];

export const getLayerByType = (type) => LAYER_REGISTRY.find((layer) => layer.type === type);

export const getLayerDefaults = (layer) => {
  if (!layer || !layer.params) return {};
  return Object.entries(layer.params).reduce((acc, [key, config]) => {
    if (config.defaultValue !== undefined) {
      acc[key] = config.defaultValue;
    } else if (config.type === "boolean") {
      acc[key] = false;
    } else {
      acc[key] = "";
    }
    return acc;
  }, {});
};

export const validateLayerParams = (layer, params = {}) => {
  const errors = {};
  if (!layer || !layer.params) return errors;

  Object.entries(layer.params).forEach(([key, config]) => {
    const value = params[key];
    const isEmpty = value === undefined || value === null || value === "";
    if (config.required && isEmpty) {
      errors[key] = `${config.label} is required`;
      return;
    }
    if (isEmpty) return;

    if (config.type === "number") {
      const num = Number(value);
      if (!Number.isFinite(num)) {
        errors[key] = `${config.label} must be a number`;
        return;
      }
      if (config.min !== undefined && num < config.min) {
        errors[key] = `${config.label} must be at least ${config.min}`;
        return;
      }
      if (config.max !== undefined && num > config.max) {
        errors[key] = `${config.label} must be at most ${config.max}`;
        return;
      }
    }

    if (config.type === "select" && Array.isArray(config.options)) {
      const allowed = config.options.map((opt) => opt.value);
      if (!allowed.includes(value)) {
        errors[key] = `${config.label} must be a valid option`;
      }
    }
  });

  return errors;
};
