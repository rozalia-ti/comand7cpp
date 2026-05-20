#include <Python.h>

#include "ai.h"

#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

#define CPPNN_PY_METHOD(func) reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(func))

struct ClassifierObject {
    PyObject_HEAD
    MlpClassifier* model;
};

bool parse_size_vector(PyObject* obj, std::vector<size_t>& output) {
    if (obj == nullptr || obj == Py_None) {
        output.clear();
        return true;
    }

    PyObject* seq = PySequence_Fast(obj, "hidden_layers must be a sequence");
    if (seq == nullptr) {
        return false;
    }

    const Py_ssize_t length = PySequence_Fast_GET_SIZE(seq);
    output.clear();
    output.reserve(static_cast<size_t>(length));

    for (Py_ssize_t i = 0; i < length; i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        const long value = PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return false;
        }
        if (value <= 0) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "hidden layer sizes must be positive");
            return false;
        }
        output.push_back(static_cast<size_t>(value));
    }

    Py_DECREF(seq);
    return true;
}

bool parse_double_vector(PyObject* obj, std::vector<double>& output) {
    PyObject* seq = PySequence_Fast(obj, "sample must be a sequence of numbers");
    if (seq == nullptr) {
        return false;
    }

    const Py_ssize_t length = PySequence_Fast_GET_SIZE(seq);
    output.clear();
    output.reserve(static_cast<size_t>(length));

    for (Py_ssize_t i = 0; i < length; i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        const double value = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return false;
        }
        output.push_back(value);
    }

    Py_DECREF(seq);
    return true;
}

bool parse_2d_double_vector(PyObject* obj, std::vector<std::vector<double>>& output) {
    PyObject* seq = PySequence_Fast(obj, "samples must be a sequence of samples");
    if (seq == nullptr) {
        return false;
    }

    const Py_ssize_t length = PySequence_Fast_GET_SIZE(seq);
    output.clear();
    output.reserve(static_cast<size_t>(length));

    for (Py_ssize_t i = 0; i < length; i++) {
        std::vector<double> row;
        if (!parse_double_vector(PySequence_Fast_GET_ITEM(seq, i), row)) {
            Py_DECREF(seq);
            return false;
        }
        output.push_back(std::move(row));
    }

    Py_DECREF(seq);
    return true;
}

bool parse_int_vector(PyObject* obj, std::vector<int>& output) {
    PyObject* seq = PySequence_Fast(obj, "targets must be a sequence of integers");
    if (seq == nullptr) {
        return false;
    }

    const Py_ssize_t length = PySequence_Fast_GET_SIZE(seq);
    output.clear();
    output.reserve(static_cast<size_t>(length));

    for (Py_ssize_t i = 0; i < length; i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        const long value = PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return false;
        }
        output.push_back(static_cast<int>(value));
    }

    Py_DECREF(seq);
    return true;
}

PyObject* convert_exception(const std::exception& exc) {
    PyErr_SetString(PyExc_RuntimeError, exc.what());
    return nullptr;
}

PyObject* Classifier_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self = reinterpret_cast<ClassifierObject*>(type->tp_alloc(type, 0));
    if (self != nullptr) {
        self->model = nullptr;
    }
    return reinterpret_cast<PyObject*>(self);
}

int Classifier_init(ClassifierObject* self, PyObject* args, PyObject* kwargs) {
    Py_ssize_t input_size = 0;
    PyObject* hidden_layers_obj = nullptr;
    unsigned int seed = 1;

    static const char* kwlist[] = {"input_size", "hidden_layers", "seed", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|OI", const_cast<char**>(kwlist), &input_size, &hidden_layers_obj, &seed)) {
        return -1;
    }
    if (input_size <= 0) {
        PyErr_SetString(PyExc_ValueError, "input_size must be positive");
        return -1;
    }

    std::vector<size_t> hidden_layers;
    if (!parse_size_vector(hidden_layers_obj, hidden_layers)) {
        return -1;
    }

    try {
        delete self->model;
        self->model = new MlpClassifier(static_cast<size_t>(input_size), hidden_layers, seed);
    } catch (const std::exception& exc) {
        convert_exception(exc);
        return -1;
    }

    return 0;
}

void Classifier_dealloc(ClassifierObject* self) {
    delete self->model;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* Classifier_input_size(ClassifierObject* self, PyObject*) {
    return PyLong_FromSize_t(self->model->input_size());
}

PyObject* Classifier_predict_proba(ClassifierObject* self, PyObject* args) {
    PyObject* sample_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &sample_obj)) {
        return nullptr;
    }

    std::vector<double> sample;
    if (!parse_double_vector(sample_obj, sample)) {
        return nullptr;
    }

    try {
        return PyFloat_FromDouble(self->model->predict_proba(sample));
    } catch (const std::exception& exc) {
        return convert_exception(exc);
    }
}

PyObject* Classifier_predict(ClassifierObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* sample_obj = nullptr;
    double threshold = 0.5;

    static const char* kwlist[] = {"sample", "threshold", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|d", const_cast<char**>(kwlist), &sample_obj, &threshold)) {
        return nullptr;
    }

    std::vector<double> sample;
    if (!parse_double_vector(sample_obj, sample)) {
        return nullptr;
    }

    try {
        return PyLong_FromLong(self->model->predict(sample, threshold));
    } catch (const std::exception& exc) {
        return convert_exception(exc);
    }
}

PyObject* Classifier_train_one(ClassifierObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* sample_obj = nullptr;
    double target = 0.0;
    double lr = 0.03;
    double sample_weight = 1.0;

    static const char* kwlist[] = {"sample", "target", "lr", "sample_weight", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Odd|d", const_cast<char**>(kwlist), &sample_obj, &target, &lr, &sample_weight)) {
        return nullptr;
    }

    std::vector<double> sample;
    if (!parse_double_vector(sample_obj, sample)) {
        return nullptr;
    }

    try {
        return PyFloat_FromDouble(self->model->train_one(sample, target, lr, sample_weight));
    } catch (const std::exception& exc) {
        return convert_exception(exc);
    }
}

PyObject* Classifier_fit(ClassifierObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* samples_obj = nullptr;
    PyObject* targets_obj = nullptr;
    unsigned long epochs = 1000;
    double lr = 0.03;
    unsigned int seed = 1;
    int shuffle = 1;
    int balance_classes = 1;

    static const char* kwlist[] = {"samples", "targets", "epochs", "lr", "seed", "shuffle", "balance_classes", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OO|kdIpp",
            const_cast<char**>(kwlist),
            &samples_obj,
            &targets_obj,
            &epochs,
            &lr,
            &seed,
            &shuffle,
            &balance_classes
        )) {
        return nullptr;
    }

    std::vector<std::vector<double>> samples;
    std::vector<int> targets;
    if (!parse_2d_double_vector(samples_obj, samples) || !parse_int_vector(targets_obj, targets)) {
        return nullptr;
    }

    try {
        self->model->fit(samples, targets, static_cast<size_t>(epochs), lr, seed, shuffle != 0, balance_classes != 0);
        Py_RETURN_NONE;
    } catch (const std::exception& exc) {
        return convert_exception(exc);
    }
}

PyObject* Classifier_predict_proba_batch(ClassifierObject* self, PyObject* args) {
    PyObject* samples_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &samples_obj)) {
        return nullptr;
    }

    std::vector<std::vector<double>> samples;
    if (!parse_2d_double_vector(samples_obj, samples)) {
        return nullptr;
    }

    try {
        const auto probabilities = self->model->predict_proba_batch(samples);
        PyObject* list = PyList_New(static_cast<Py_ssize_t>(probabilities.size()));
        if (list == nullptr) {
            return nullptr;
        }
        for (size_t i = 0; i < probabilities.size(); i++) {
            PyObject* value = PyFloat_FromDouble(probabilities[i]);
            if (value == nullptr) {
                Py_DECREF(list);
                return nullptr;
            }
            PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), value);
        }
        return list;
    } catch (const std::exception& exc) {
        return convert_exception(exc);
    }
}

PyMethodDef Classifier_methods[] = {
    {"input_size", reinterpret_cast<PyCFunction>(Classifier_input_size), METH_NOARGS, "Return model input size."},
    {"predict_proba", reinterpret_cast<PyCFunction>(Classifier_predict_proba), METH_VARARGS, "Return probability of class 1."},
    {"predict", CPPNN_PY_METHOD(Classifier_predict), METH_VARARGS | METH_KEYWORDS, "Return predicted class."},
    {"train_one", CPPNN_PY_METHOD(Classifier_train_one), METH_VARARGS | METH_KEYWORDS, "Train on one sample."},
    {"fit", CPPNN_PY_METHOD(Classifier_fit), METH_VARARGS | METH_KEYWORDS, "Train on a batch of samples."},
    {"predict_proba_batch", reinterpret_cast<PyCFunction>(Classifier_predict_proba_batch), METH_VARARGS, "Return probabilities for samples."},
    {nullptr, nullptr, 0, nullptr}
};

PyTypeObject ClassifierType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
};

PyModuleDef cppnn_module = {
    PyModuleDef_HEAD_INIT,
    "cppnn",
    "C++ neural network models.",
    -1,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

} // namespace

PyMODINIT_FUNC PyInit_cppnn() {
    ClassifierType.tp_name = "cppnn.MlpClassifier";
    ClassifierType.tp_basicsize = sizeof(ClassifierObject);
    ClassifierType.tp_itemsize = 0;
    ClassifierType.tp_dealloc = reinterpret_cast<destructor>(Classifier_dealloc);
    ClassifierType.tp_flags = Py_TPFLAGS_DEFAULT;
    ClassifierType.tp_doc = "Small C++ MLP classifier for binary classification.";
    ClassifierType.tp_methods = Classifier_methods;
    ClassifierType.tp_init = reinterpret_cast<initproc>(Classifier_init);
    ClassifierType.tp_new = Classifier_new;

    if (PyType_Ready(&ClassifierType) < 0) {
        return nullptr;
    }

    PyObject* module = PyModule_Create(&cppnn_module);
    if (module == nullptr) {
        return nullptr;
    }

    Py_INCREF(&ClassifierType);
    if (PyModule_AddObject(module, "MlpClassifier", reinterpret_cast<PyObject*>(&ClassifierType)) < 0) {
        Py_DECREF(&ClassifierType);
        Py_DECREF(module);
        return nullptr;
    }

    return module;
}

#undef CPPNN_PY_METHOD
